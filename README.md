# VoiceDesk — AI Phone Receptionist

Answers a business phone line, understands the caller, books or reschedules appointments in a real calendar, and hands off to a human when it should. Built for clinics, salons, and trades businesses that lose revenue to missed calls.

**Why this exists:** a missed call is a lost booking. The hard part is not speech — it's knowing when to stop talking and escalate.

## v1 Scope

- Inbound call answered, greeting played
- Streaming speech-to-text, LLM turn, text-to-speech reply
- Tools: `check_availability`, `book_appointment`, `reschedule`, `cancel`, `take_message`, `transfer_to_human`
- Appointments written to a real calendar store (Postgres, with an interface ready for Google Calendar)
- Full transcript, tool calls, latency, and cost logged per call
- Post-call SMS/email confirmation to the caller
- Browser-based test harness (mic in the browser) so you can develop without spending phone credits

## Non-goals (do not build)

- Outbound calling or any dialler
- Payment capture over the phone
- Multi-tenant admin panel or billing
- Voice cloning or custom TTS training
- Interrupt-free "perfect" barge-in tuning. Basic barge-in only.

## Stack

- Python 3.13, FastAPI, Uvicorn, WebSockets
- SQLAlchemy 2.x, Alembic, psycopg 3, PostgreSQL
- Telephony: Twilio Media Streams (trial credit is enough for v1)
- STT and TTS behind provider interfaces — start on whatever free tier is available, swap without touching the dialogue code
- Anthropic SDK for the dialogue turn, with tool use
- pytest, HTTPX

Develop against the browser harness first. Telephony is milestone 6, not milestone 1. This keeps cost near zero while the logic is unstable.

## Architecture

```
caller → Twilio Media Stream (WS) → audio buffer
       → STT (streaming) → turn manager
       → LLM (tools) → tool executor → calendar
       → TTS → audio back to caller
       → transcript + metrics persisted
```

Layout:

- `app/telephony/` — Twilio webhook + media stream socket
- `app/audio/` — buffering, resampling, VAD, barge-in
- `app/providers/` — `stt.py`, `tts.py`, `llm.py` interfaces plus implementations
- `app/dialogue/` — turn manager, system prompt, tool definitions
- `app/tools/` — one module per tool, each independently testable without audio
- `app/calendar/` — availability logic, booking, conflict prevention

The dialogue layer must be testable with text in, text out, no audio at all. If testing a booking flow requires a phone call, the layering is wrong.

## Data model

- **calls** — id, direction, from_number, to_number, started_at, ended_at, outcome (`booked|rescheduled|cancelled|message_taken|transferred|abandoned|failed`), total_cost_usd
- **turns** — id, call_id FK, role (`caller|agent`), text, audio_ms, stt_latency_ms, llm_latency_ms, tts_latency_ms, created_at
- **tool_calls** — id, turn_id FK, tool_name, arguments (JSONB), result (JSONB), success, error, latency_ms
- **appointments** — id, customer_name, phone, service_id, starts_at, ends_at, status, call_id FK, created_at
- **services** — id, name, duration_minutes, staff_id, active
- **business_hours** — id, weekday, opens_at, closes_at

## Behaviour rules (encode in the system prompt and test them)

- Never invent availability. Availability comes only from `check_availability`.
- Confirm name, service, date and time back to the caller before calling `book_appointment`.
- Escalate to `transfer_to_human` on: medical or legal advice requests, complaints, refunds, anything asked twice without resolution, or explicit "let me speak to someone".
- If STT confidence is low twice in a row, offer to take a callback number instead of guessing.
- Keep replies under roughly 2 sentences. Long replies destroy call feel.
- Never state a price unless it exists in the services table.

## Latency budget

Target under 1.2s from caller stopping to agent starting to speak.

- Stream STT; do not wait for the full utterance
- Start the LLM call on endpoint detection, not on silence timeout
- Stream TTS and start playback on the first audio chunk
- Log all three latencies per turn; the README reports p50 and p95

## Build milestones

1. FastAPI skeleton, config, `/health`, Postgres + Alembic, `services` and `business_hours` tables
2. Calendar core: availability calculation, conflict-safe booking, unit tests. No AI yet.
3. Tool layer: all six tools callable from tests as plain functions
4. Dialogue layer: text-in/text-out conversation with tool use, transcripts persisted
5. Browser harness: mic → WS → STT → dialogue → TTS → speaker
6. Twilio inbound: webhook, media stream, same dialogue path
7. Barge-in + latency instrumentation
8. Post-call confirmation (SMS or email) and call summary
9. Scenario eval suite
10. README with results, demo recording, deploy notes

## Evals

`evals/scenarios/` — at least 15 written call scenarios as caller-turn scripts, run against the text-level dialogue layer in CI:

- straightforward booking
- caller changes their mind mid-booking
- requested slot unavailable
- reschedule an existing appointment
- caller is angry → must transfer
- caller asks a medical question → must transfer
- caller gives a partial or ambiguous date ("next Tuesday-ish")
- background noise / garbled turn twice → must offer callback

Metrics: task success rate, wrong-tool-call rate, hallucinated-availability rate (must be zero), escalation precision and recall, mean turns to booking, cost per call.

**Acceptance for v1:** ≥85% task success across scenarios, zero hallucinated availability, correct escalation on all four escalation scenarios.

## Cost tracking

Log STT seconds, TTS characters, LLM tokens, telephony minutes per call, and compute `total_cost_usd`. README must state cost per call and cost per completed booking. That single number is the sales argument.

## Notes for the implementer

- Build milestones 1–4 with no audio involved at all. Audio is an adapter, not the product.
- Double-booking is the failure that ends a client relationship. Enforce it with a database constraint, not application logic alone.
- Store every transcript. Redact nothing in v1, but keep PII fields in named columns so redaction is easy to add later.
- Small, reviewable changes. Do not introduce a realtime voice SDK that replaces the provider interfaces without being asked.

---

## Implementation status

**Milestones 1 and 2 are implemented. Milestones 3–10 are not started.** Everything
above this line is the specification; everything below describes only what
exists today.

### What milestone 1 built

- FastAPI application (`app/main.py`) serving exactly one endpoint, `GET /health`.
- Environment-driven configuration (`app/config.py`, prefix `VOICEDESK_`) —
  application identity and the database URL, nothing more.
- SQLAlchemy 2.x on psycopg 3 (`app/db/`), lazily-created cached engine.
- All six tables from the data model, as ORM models under `app/models/`:
  `calls`, `turns`, `tool_calls`, `appointments`, `services`, `business_hours`.
- One Alembic migration creating them, verified to upgrade, downgrade and
  re-upgrade on a fresh database.
- Tests exercising the schema against a real migrated PostgreSQL database
  rather than in memory.

### The double-booking constraint

The specification says double booking must be prevented "with a database
constraint, not application logic alone". That constraint exists now, ahead of
any booking code, because milestone 2 needs it to already be true:

```sql
EXCLUDE USING gist (
    staff_id WITH =,
    tstzrange(starts_at, ends_at, '[)') WITH &&
) WHERE (status = 'booked')
```

Two callers can both pass an availability check before either commits; only the
database can settle that race. Three consequences, each covered by a test:

- **`appointments.staff_id` is denormalised from the service.** An exclusion
  constraint can only reference its own table's columns, and what must not
  overlap is one person's diary — a stylist offering two services must not be
  booked for both at once. Keeping it in step with `services.staff_id` is
  milestone 2's job.
- **The range is half-open**, so 09:00–09:30 and 09:30–10:00 do not collide.
- **The constraint is partial**, so a cancelled appointment stops holding its slot.

It needs the `btree_gist` extension, which the migration installs (trusted
since PostgreSQL 13, so no superuser is required).

### What milestone 2 built

The calendar core, in `app/calendar/` — deterministic business logic with no
AI anywhere near it:

| Module | Responsibility |
| --- | --- |
| `hours.py` | Turns wall-clock opening times into instants; does an interval fit one opening period? |
| `availability.py` | Candidate slots from the grid, minus that staff member's booked intervals |
| `service.py` | `CalendarService`: `available_slots`, `is_available`, `book`, `reschedule`, `cancel` |
| `errors.py` | Eight domain errors, naming business situations rather than SQL ones |

The boundary is deliberate:

```
future tool / dialogue layer
          ↓
    CalendarService          ← availability is computed here, never guessed
          ↓
      PostgreSQL             ← the authority on conflicts
```

**Availability is advisory; the database decides.** `book` and `reschedule` do
not check-then-insert — a check taken a moment earlier can already be stale.
They write, and turn the exclusion constraint's refusal into `SlotUnavailable`,
inside a savepoint so the session survives. A test runs two sessions that both
check the same slot, both are told it is free, and proves exactly one booking
survives.

Behaviour worth knowing:

- Intervals are half-open throughout, matching the constraint: 10:00–10:30 and
  10:30–11:00 coexist; 10:00–10:30 and 10:15–10:45 do not.
- An appointment must fit inside **one** opening period — it may not run
  through a lunch break, even though both sides are open.
- Starting exactly at opening is inside; ending exactly at closing is inside;
  starting exactly at closing is not.
- Availability follows the *staff member*, not the service: one person's two
  services block each other.
- Rescheduling updates the row in place, so the appointment keeps its id,
  creation time and originating call — a moved booking is the same commitment
  at a new time.
- Cancelling marks the row rather than deleting it, and the partial constraint
  means the slot is immediately bookable again.

### Assumptions recorded in milestone 2

The specification defines neither of these, so both are configuration with
documented defaults rather than invented schema:

- **`VOICEDESK_BUSINESS_TIMEZONE`** (default `UTC`). `business_hours` holds
  wall-clock times and appointments are instants; something must say which
  wall clock. VoiceDesk serves one business — multi-tenancy is a non-goal — so
  this is one setting, not a column. A deployment sets its own. Opening periods
  on a daylight-saving transition day are resolved by `zoneinfo`'s default fold
  handling and by no rule of ours; with the default `UTC` there are no
  transitions.
- **`VOICEDESK_SLOT_GRANULARITY_MINUTES`** (default `15`). The grid available
  start times sit on, measured from each period's own opening time rather than
  from midnight, so a shop opening at 09:10 offers 09:10, 09:25, … .

`services.staff_id` is singular, so a service has exactly one staff member and
no staff-selection or load-balancing logic exists.

Milestone 2 required **no schema change and no migration**: the M1 tables and
the exclusion constraint already supported everything.

### Deliberately not built yet

No audio, STT, TTS, WebSockets, Twilio, Anthropic calls, tool layer, dialogue
layer, transfer, SMS, email, barge-in, cost computation, scenario evals or
deployment. The `app/telephony/`, `app/audio/`, `app/providers/`,
`app/dialogue/` and `app/tools/` packages from the layout above **do not
exist** — they will be created by the milestones that need them, rather than
standing empty.

There is **no booking API**: the calendar core is a library the milestone-3
tool layer will call. The only HTTP endpoint remains `GET /health`.

`AppointmentStatus` has two values, `booked` and `cancelled`: `reschedule`
moves an existing booking's times and leaves it `booked`, so it is not a third
state. `calls.direction` allows `outbound` because a direction has two values,
but v1 only ever writes `inbound` — outbound calling is a non-goal.

### Running locally (milestone 1)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

createuser voicedesk --pwprompt
createdb -O voicedesk voicedesk
createdb -O voicedesk voicedesk_test     # the test suite migrates and drops this

alembic upgrade head
uvicorn app.main:app --reload
curl localhost:8000/health
pytest
```

Alembic reads the database URL from `VOICEDESK_DATABASE_URL` via
`app/config.py`; `alembic.ini` deliberately holds no URL, so migrations and the
app cannot disagree about which database they are using. Tests that need
PostgreSQL are skipped when no server answers, so `pytest` still runs without
one (16 pass, 99 skip). With a database: 115 pass.

VoiceDesk is a separate application from DocIntel in this repository: its own
package, dependencies, virtualenv, configuration prefix and database. Nothing
is shared between them.
