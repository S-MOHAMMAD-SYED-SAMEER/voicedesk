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

**Milestones 1 to 8 are implemented. Milestones 9 and 10 are not started.**
Milestone 8 here is the specification's *Cost tracking* section; post-call
SMS or email confirmation is not built. Everything
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

### What milestone 3 built

The tool layer: the six tools from the scope list, in `app/tools/`, one module
each. Every one is a plain function with the same shape —

```python
def book_appointment(context: ToolContext, *, service_name, starts_at,
                     customer_name, phone) -> ToolResult: ...
```

— so a test calls it with a database session and nothing else. No audio, no
telephony, no model, no HTTP endpoint. `GET /health` is still the only route.

**Tools delegate; they do not reimplement.** Availability, opening hours,
durations and conflict handling live in `app/calendar/` and are reached
through `CalendarService`. A tool normalises its arguments, calls the
service, and describes what happened. `tests/test_tools_smoke.py` enforces
this by parsing each tool module: none may mention `BusinessHours`,
`tstzrange`, `select(Appointment`, `timedelta`, `duration_minutes` or the
calendar's internal helpers, and none may import a model provider or
telephony library. Two answers to "is this free?" would be one too many, and
the second one would be the one that hallucinated.

**Failures are returned, not raised.** `ToolResult` is `success`, `data` and
`error` — deliberately the shape of `tool_calls.result`, `tool_calls.success`
and `tool_calls.error`, so milestone 4 can write one straight into the other.
A model that asks for a slot someone else has just taken has to be told so in
a way it can recover from; an exception would end the turn instead. The losing
side of a race gets `success=False` with `slot_taken=True`, and its session
stays usable — there is a test for exactly that.

**The call is ambient, not an argument.** `ToolContext` carries the session,
the settings and the `call_id`. No tool accepts a `call_id`, so a model cannot
attribute a booking to a call that is not its own.

**`TOOLS` is the single registry.** It maps the six specified names to the
six functions; `get_tool` looks one up, and an unknown name raises rather than
returning a business failure. Milestone 4 builds its tool definitions from
this dictionary rather than keeping a second list that can drift.

### Decisions recorded in milestone 3

Five points the specification left open, resolved here rather than invented
later:

* **Tools take a `service_name`, not a service id.** A caller says "a
  haircut"; ids are the calendar's currency, not a conversation's. Resolution
  is deterministic and never guesses: trimmed, case-insensitive, exact, and
  against **active** services only. Exactly one match proceeds; zero matches
  fails and lists what is actually offered; two or more fails as ambiguous.
  There is no fuzzy matching, so "Haircuts" does not silently become
  "Haircut".
* **Availability comes back as ISO-8601 strings in the configured business
  timezone**, with that timezone named in the result. An empty list is a
  *successful* answer — "nothing free that day" is information, not an error —
  so a model can tell it apart from a lookup that went wrong.
* **A timestamp with no offset is read as wall-clock time in the business
  timezone**, the mirror of how availability is reported. "Ten o'clock" from a
  caller means ten where the business is, and a naive timestamp names no
  instant on its own. `CalendarService` still refuses naive datetimes; the
  tool layer is where a caller's words become an instant.
* **`take_message` returns the message; it does not store it.** Nothing in
  milestone 3 persists anything of its own. `tool_calls.turn_id` is `NOT
  NULL`, and turns are the dialogue layer's to create, so every tool's
  arguments and result are written once, by milestone 4, through `tool_calls`.
  A messages table now would give the same message two homes.
* **`transfer_to_human` records the escalation and its reason; it does not
  connect anything.** Moving audio is telephony's job and belongs to the
  Twilio milestone. The reason is required, because "escalation precision and
  recall" cannot be measured against an escalation that never said why.

No schema change was needed, so there is still exactly one migration:
`alembic check` reports no new operations.

### What milestone 4 built

The dialogue layer: text in, text out, with tool use in between. No audio, no
telephony, no HTTP endpoint — `GET /health` is still the only route. A whole
booking conversation is a unit test.

```
caller text → Conversation → LanguageModel → tool_use
                                  ↓
                           ToolExecutor → app.tools registry
                                  ↓
                         CalendarService → PostgreSQL
                                  ↓
                          tool_result → LanguageModel → reply
```

**One module imports the SDK.** `app/providers/llm.py` defines a vendor-neutral
`LanguageModel` protocol and its dataclasses; `app/providers/anthropic_llm.py`
is the only file in the project that says `import anthropic`, and it takes an
injectable client. So the dialogue layer, the tool layer and all 352 tests run
with no API key and no network — verified by running the suite with the key
unset and the SDK's base URL pointed at a dead port. A source-parsing test
enforces the boundary.

**The tools the model sees are generated from the tools that exist.**
`app/dialogue/definitions.py` builds the six schemas by walking `app.tools.TOOLS`
and comparing each schema against the real function signature. A tool added to
the registry without a schema, a schema for a tool that is not registered, or
an argument that drifts, all raise rather than reach a model as a quiet lie.
Schemas are strict: closed objects, every argument required. They describe
argument shapes and nothing about durations, opening hours or availability.

**The prompt asks; the application decides.** Every behaviour rule the
specification lists is in `app/dialogue/prompt.py`, and every one of them that
can be enforced in code is *also* enforced in code. The prompt is a constant
with no timestamps or identifiers, so the same conversation produces the same
bytes; the only thing read from the database is the list of active service
names, because a model with no menu has to guess at one.

**The loop always ends.** One caller turn goes round the model/tool loop at
most `max_tool_iterations` times (default 8). Running out is a failure with a
fixed reply, not an answer. So is a model that cannot be reached. In both cases
`DialogueResult.failed` is true, the caller hears something safe, and the
partial transcript is still written — the call that went wrong is the one
somebody will want to read.

### The booking guard

The specification's acceptance criterion is a hallucinated-availability rate of
**zero**. A prompt can ask for that; only code can hold it. So the executor
keeps an in-memory ledger of the slots the calendar has actually offered during
this conversation, and:

> `book_appointment` runs only for a start time that a **successful**
> `check_availability` returned earlier in the same conversation.

Anything else is refused before the tool is reached, returned to the model as a
structured failure it can recover from, and persisted as a failed call. The
comparison is on instants rather than strings, so the same moment written with
a different offset still matches, and service names are compared the way the
tool layer resolves them. A failed or empty availability check verifies
nothing, and one conversation's ledger cannot verify another's booking.

Three things the guard deliberately does **not** do:

* It does not decide whether the *caller* agreed. That is a conversation, not a
  fact, and it stays a behavioural rule tested by the milestone-9 scenarios.
* It does not stop the model *saying* a wrong time out loud. Only tool calls
  pass through here.
* It does not decide whether a verified slot is still free. `CalendarService`
  and the database's exclusion constraint remain the authority on that, and a
  race lost between the check and the write comes back as `slot_taken` — there
  is a test that stages exactly that interleaving.

`reschedule` is not guarded. The calendar already refuses a time outside
opening hours or one that is taken, and guarding it would mean requiring an
availability check the caller never asked for.

### Decisions recorded in milestone 4

* **No schema change, and no migration.** `turns` and `tool_calls` already had
  every column milestone 4 needs. `alembic check` reports no new operations and
  there is still one migration file.
* **One caller turn and one agent turn per `send()`**, whatever happened in
  between. A row per model response would leave empty-text turns behind, and
  `turns` is what milestone 9's "mean turns to booking" counts.
  `llm_latency_ms` on the agent turn is the summed model latency for the turn.
* **Both `created_at` values are set in Python.** PostgreSQL's `now()` is the
  *transaction's* time, so two rows written together would share it — and
  `Call.turns` orders by `created_at`.
* **A failed tool call stores a null result.** M3 failures carry recovery data
  (`services_offered`, `slot_taken`) and that data *is* returned to the model
  in the `tool_result` block, because it is how the model recovers. It is not
  an outcome, so it is not written to `tool_calls.result`.
* **Conversation history lives in memory, not in the database.** It contains
  the provider's tool-use and tool-result blocks; `turns` stores text. Making
  the database round-trip an API transcript would mean columns milestone 4 does
  not justify. `turns` and `tool_calls` are the readable audit record.
* **Anthropic's `tool_use` ids are never persisted.** They correlate a request
  with its response and mean nothing afterwards.
* **`Conversation` is given a `Call`; it never invents one.** `calls.from_number`
  and `to_number` are `NOT NULL` and milestone 4 is text-only, so what a
  browser-harness call puts there is milestone 5's question.
* **Adaptive thinking, effort `low`.** The budgeted form of extended thinking
  is removed on current models. Effort is the latency lever, and the
  specification's budget is 1.2 seconds. Thinking is deliberately left *on*:
  with it disabled the model can write a tool call into visible text instead of
  calling the tool, which for a receptionist means silently never booking.
* **There is no price column, so there are no prices.** The specification says
  never to state a price unless it is in the services table; the table holds
  none, so the prompt says plainly that the system has no pricing and routes
  the question to a message or a human. Adding the column would be a schema
  change nothing has asked for.
* **`SYSTEM_PROMPT_VERSION` is a code constant** (`m4.1`), reported on every
  `DialogueResult` and stored nowhere.

Known limitation: `tool_calls` has no ordering column, so the order of several
tool calls within one turn is not recoverable from the database alone. The
transcript and the model's own history carry it; adding a column would be a
migration, and nothing yet needs one.

### What milestone 5 built

The audio layer, wrapped around the dialogue layer rather than through it:

```
browser mic → WAV → WS /ws/harness → VoiceSession
                                          ↓
                            speech-to-text → Conversation → text-to-speech
                                          ↓
                        WS: JSON turn, then the reply as a WAV → speaker
```

`VoiceSession` (`app/audio/session.py`) validates the audio, transcribes it,
hands the text to milestone 4, and turns the answer back into sound. It holds
no conversation state, runs no tools, touches no calendar and never speaks to
a model — "audio is an adapter, not the product". A source-parsing test
enforces every one of those.

**It is utterance-based, and the latency target is not claimed.** Press and
hold to talk, release to send one complete recording; the reply comes back as
one complete recording. There is no streaming speech-to-text, no streaming
synthesis, no voice-activity detection, no endpoint detection and no barge-in.
The specification's 1.2-second target is therefore **neither met nor measured
in milestone 5** — press-to-talk cannot meet it, and no number in this
repository suggests otherwise. Streaming and barge-in are milestone 7's, and
the provider protocols are shaped so that milestone adds a method rather than
replacing one.

**One audio format, checked at the door.** 16 kHz, mono, signed 16-bit
little-endian PCM in a WAV container, both directions. Chosen because `wave`
and `struct` handle it with no dependency at all, because a WAV describes
itself so nothing has to be told its sample rate out of band, and because it
is what every transcriber accepts. Anything else — a non-WAV payload, stereo,
8 kHz, 24-bit, or more than 1 MiB — is rejected before a provider is asked to
make sense of it. It is not a telephony format: that is 8 kHz µ-law, and
`AudioFormat.encoding` is a string precisely so milestone 6 can say so.
(Worth knowing in advance: Python 3.13 removed `audioop`, so that conversion
will have to be written rather than imported.)

**Offline by default.** `stt_provider` and `tts_provider` default to
`offline`, so a fresh clone runs the entire harness — microphone, dialogue,
booking, speaker — with no account, no key, no network and no bill. Be clear
about what those are: `offline_stt.py` returns **a fixed sentence whatever you
say**, and `offline_tts.py` returns **a tone, not a voice**. They are a
harness. Neither is speech recognition and neither is speech.

**Two real adapters, neither ever called for real.** `deepgram_stt.py` and
`elevenlabs_tts.py` are plain REST over `httpx` — no vendor SDK, and each is
the only module in the project that knows its vendor's endpoint or header
names, which a test asserts. Their request shapes are verified against a mock
transport; **no request has ever left this repository**, so no accuracy,
latency or cost claim is made about either service. Set a key and switch the
provider setting to use them.

### The browser harness

`GET /harness` serves one page: vanilla JavaScript, no framework, no npm, no
build step. It captures microphone audio, converts it to 16-bit PCM at 16 kHz,
builds a WAV and sends it as one binary frame. It is a development tool, not a
frontend.

`WS /ws/harness` is the call. A WebSocket rather than a POST because a
`Conversation` holds its history in memory, so the connection *is* the call:
opening one starts it, closing one ends it, and nothing needs a session
registry keyed by an identifier the browser could get wrong.

The server always sends **the JSON turn first and the audio second**, so the
page never has to guess what a binary frame belongs to. A rejected frame gets
an `error` message and the call carries on; a frame that fails unexpectedly —
a provider with no credentials, say — is reported the same way rather than
dropping the socket, which would look like a bug in the browser.

### Decisions recorded in milestone 5

* **No schema change, and no migration.** Still one migration file, and
  `alembic check` reports no new operations.
* **A browser call uses sentinels, not invented numbers.**
  `calls.from_number = "browser"` and `to_number = "harness"`. Those columns
  are `NOT NULL` and a browser has neither; a plausible-looking fake number
  would eventually be read as a real one, and making the columns nullable
  would be a migration to accommodate a development tool.
* **`calls.outcome` is deliberately left null.** Summarising how a call went
  belongs to the milestone that owns post-call reporting. A guess written now
  would be indistinguishable from a fact later. Milestone 5 manages
  `started_at` and `ended_at` only, and if the process dies before hanging up,
  `ended_at` stays null — nobody hung up.
* **`turns.audio_ms`, `stt_latency_ms` and `tts_latency_ms` stay null.** They
  are reported on `VoiceTurn` and to the browser, and written nowhere.
  Milestone 4 commits its turn rows before synthesis runs and does not hand
  them back, so filling these would mean either changing the dialogue layer or
  reaching back into rows it had already written. Durable latency belongs to
  the milestone that owns observability; an empty column is more honest than
  one filled by a layer that had to go behind another's back.
* **Confidence is carried and not acted on.** `Transcript.confidence` reaches
  `VoiceTurn` and the browser. The specification's rule — low confidence twice
  in a row, offer a callback — needs endpoint detection to mean anything, so
  it belongs with the milestone that builds that. **Milestone 5 does not
  implement it**, and says so rather than half-building it.
* **The greeting is not a turn.** Picking up the phone is synthesised and
  played, but nobody said anything to prompt it: no model call, no `turns`
  rows, no `tool_calls` row.
* **Three failures, three answers.** Nothing heard: the model is *not called
  at all* — silence is not a question, and asking a model about it invites
  invention. The transcriber failed: the model is not called and no transcript
  is fabricated, because a broken provider must never look like silence. The
  synthesiser failed: the dialogue already happened and is already persisted,
  including any booking it made, so it is **not re-run** — retrying for audio
  would risk booking the same caller twice. The reply comes back as text with
  `speech=None`.
* **A failed dialogue is spoken verbatim.** Milestone 4 already returns
  something safe; this layer does not add a second apology or a different
  outcome.

### What milestone 6 built

The telephony adapter, beside the browser one rather than instead of it:

```
caller ── PSTN ──► carrier
                     │
                     ├─ POST /telephony/voice ─► signature ─► Call row ─► TwiML
                     │
                     └─ WS /telephony/stream
                              │  connected / start / media / stop
                              ▼
                     µ-law 8 kHz ──► app/audio/telephony.py ──► WAV 16 kHz
                              │
                              ▼
                     VoiceSession ─► Conversation ─► tools ─► calendar ─► PG
                              │
                     WAV 16 kHz ──► app/audio/telephony.py ──► µ-law frames
                              ▼
                           carrier ──► caller
```

```
browser adapter   (app/api/harness.py)      ──┐
                                              ├──► VoiceSession ──► Conversation
telephony adapter (app/telephony/stream.py) ──┘
```

**There is no second dialogue.** `VoiceSession.speak(Audio)` is the same call
the browser harness makes; only the audio at either end differs. Nothing in
`app/telephony/` imports the calendar, a tool, or a model implementation, and
a source-parsing test enforces each of those. No interface anywhere in
milestones 1 to 5 changed.

**No carrier SDK.** Three things are needed from the carrier and all three are
standard library: read its JSON (`json`), answer with TwiML (a string), and
check its signature (`hmac`, `hashlib`, `base64`). Nothing was added to
`pyproject.toml` at all.

### Telephone audio

Telephony is G.711 µ-law: 8 kHz, mono, one byte a sample. Inside VoiceDesk
everything is 16 kHz mono 16-bit PCM in a WAV container. `app/audio/telephony.py`
is the whole of the conversion, and it knows nothing about who is carrying the
audio.

**The codec is written, not imported.** Python 3.13 removed `audioop`
(PEP 594), and a backport, NumPy or ffmpeg would all be out of proportion to a
256-value codec. It is checked against the standard's anchors: silence encodes
to `0xFF`, `0xFF` decodes to zero, and the extremes clip to `0x80` and `0x00`.
One detail worth knowing rather than discovering: **255 of 256 byte values
survive a decode-then-encode round trip, not 256.** `0x7F` is *negative* zero;
it decodes to zero, which re-encodes to the positive-zero code `0xFF`. That is
the standard having two codes for zero, not a defect, and the test says so
explicitly rather than demanding a round trip that cannot exist.

**The resampler is deliberately simple, and makes nothing sound better.**
Upsampling interpolates between neighbours; downsampling averages pairs. That
is a crude anti-alias filter, and telephone audio is band-limited to about
3.4 kHz whatever is done to it — going to 16 kHz invents no detail that 8 kHz
did not carry. **No claim is made that this improves transcription.**

Outbound audio is cut into 160-byte frames, which is 20 ms at 8 kHz, and a
`mark` follows each batch so the carrier can say when it finished playing.
Frames are not paced: pacing so playback can be interrupted is barge-in.

### The temporary utterance boundary

A carrier streams continuously; the dialogue layer wants complete utterances.
The boundary used here is an amplitude timer — buffer once something is above
`telephony_silence_threshold`, close the utterance after `telephony_silence_ms`
below it, and close it anyway at `telephony_max_utterance_ms`, which is also
what bounds the buffer on an open socket.

**This is not voice-activity detection.** There is no spectral analysis, no
adaptive noise floor, no speech classifier, no endpoint classifier, no partial
transcripts and no barge-in. It is the least mechanism that turns a continuous
stream into turns, it is temporary, and **the milestone that owns real-time
behaviour replaces it entirely.** A caller who says nothing produces no
utterance and no model call at all.

### Running a real phone line

```bash
export VOICEDESK_TELEPHONY_ENABLED=true
export VOICEDESK_TWILIO_AUTH_TOKEN=...        # also the signature key
export VOICEDESK_PUBLIC_BASE_URL=https://your-tunnel.example.com
```

Then point the number's voice webhook at `POST {public_base_url}/telephony/voice`.
The public URL is required and cannot be inferred: it is both the address in
the TwiML stream URL and the address the signature is checked against, and
behind a proxy the request describes the hop rather than the address the
carrier dialled. While developing, an ngrok or Cloudflare tunnel is enough.

With `telephony_enabled` false — the default — `POST /telephony/voice` answers
404 and the media socket closes immediately. Nothing telephonic is reachable
until somebody turns it on.

**Security.** The webhook signature is checked before anything is written, so
an unsigned request creates no row and costs nothing. The media socket is not
signed by the carrier, so it authenticates by binding to a call the *signed*
webhook created: a stream naming a call nobody answered is refused and closed,
and no row is invented for it. Replay windows, rate limiting and IP
allow-listing are production hardening and are not implemented.

### What has and has not actually been proved

The automated tests and the simulated smoke test drive a **real** WebSocket
against a **real** application and a **real** PostgreSQL database, with
scripted model and speech providers. They exercise the webhook, the signature,
the µ-law conversion, the utterance boundary, the dialogue path and the call
lifecycle.

**They do not place a telephone call.** No Twilio account, no phone number and
no PSTN traffic has been involved at any point. Nothing here establishes real
carrier behaviour, real latency, real transcription accuracy on telephone
audio, real voice quality, or production readiness, and no such claim is made
anywhere in this repository.

### Decisions recorded in milestone 6

* **One migration, the first since milestone 1.** `calls.provider_call_sid`,
  `String(64)`, nullable and unique. It is the only durable link between a
  VoiceDesk row and the carrier's record of the same call — what a billing
  question or a support ticket is keyed by — and the unique constraint is what
  makes a retried webhook safe, better than application code remembering to
  look first. Nullable because a browser call has no carrier; verified safe to
  apply to a table that already has rows.
* **The browser sentinels are unchanged.** `from_number="browser"` /
  `to_number="harness"` with a null `provider_call_sid`. Two adapters write
  `calls`, and neither touches the other's rows.
* **The webhook creates the call; the socket finds it.** The stream looks the
  call up by `provider_call_sid` rather than keeping an in-memory map, which
  also means the webhook and the socket need not land on the same worker.
* **`calls.outcome` and `total_cost_usd` stay null**, and so do
  `turns.audio_ms`, `stt_latency_ms` and `tts_latency_ms`. Summarising a call
  and accounting for it belong to the milestone that owns post-call reporting;
  durable latency belongs to the one that owns observability.
* **The greeting is not a turn.** Synthesised through the configured provider
  after `start`, played down the line, and recorded nowhere: no model call, no
  `turns` row, no `tool_calls` row. Deliberately not the carrier's `<Say>`,
  which would put a second unrelated voice on the line.
* **The dialogue runs on a worker thread.** `VoiceSession.speak` is
  synchronous all the way down — speech recognition, the model, tool calls,
  PostgreSQL, synthesis — and running it inline would block the event loop for
  seconds while the carrier kept sending. `anyio.to_thread.run_sync` keeps the
  socket responsive without making anything in milestones 4 or 5 async.
* **One malformed frame never ends a call.** Unparsable JSON, a missing field,
  a payload that is not base64, an event this milestone has never heard of —
  each is logged and stepped over. A carrier is entitled to add an event, and
  hanging up on the caller over it would be the worse bug. What *is* refused
  is audio in a format that would only decode to noise, and a stream naming an
  unknown call.
* **Our own voice is never transcribed.** Only the inbound track is buffered;
  feeding the outbound track back to speech recognition would be a loop.

### What milestone 7 built

Realtime voice: the receptionist listens while it is talking, and stops when
you interrupt it.

```
frames ─► VAD ─► endpointer ─► streaming STT ─► Conversation ─► streaming TTS ─► sink
             │                      partials → display only            ▲
             └── speech during playback ── cancel, drain, clear ────────┘
```

**The reader never waits for a turn.** This is the structural change the whole
milestone rests on. Milestone 6 awaited each turn inline, which meant no
inbound audio was read for its duration — a receptionist that cannot hear you
while it is thinking cannot be interrupted. Turns now run in a task beside the
reader, and a frame is accounted for the moment it arrives.

**One dialogue, two sessions.** `RealtimeSession` is the streaming
counterpart to `VoiceSession`, which is unchanged and still serves
press-to-talk. Both call `Conversation.send`, so there is one dialogue
implementation, one tool registry and one calendar beneath them. Nothing in
`app/realtime/` imports the calendar, a tool, a provider implementation, or
any transport — source-parsing tests enforce each of those.

**Nothing was added to `pyproject.toml` but a declaration.** `anyio` was
already installed by FastAPI and is now declared because application code
imports it. No VAD library, no NumPy, no Torch, no ONNX, no vendor SDK.

With `realtime_enabled` off — the default — none of this runs and the
milestone-6 path does: one complete utterance, one complete reply.

### Voice activity, and what it is not

`app/audio/vad.py` measures a frame's RMS energy and how often it crosses
zero, against a noise floor that re-learns itself from quiet frames, with
hysteresis so one loud frame does not start an utterance and one quiet frame
does not end one.

**It is an energy detector, and it can be fooled.** Sustained background noise
louder than the margin reads as somebody talking — a television in the next
room will hold the line open, and because speech never teaches the floor, it
will keep doing so. A quietly spoken word on a noisy line can be missed
entirely. There is a test that asserts exactly this rather than hiding it.
**No claim of parity with a trained detector is made**, and if that matters
for a deployment, the interface is small enough to put one behind.

What it buys instead: a few dozen lines of standard library running about six
hundred times faster than real time, every decision reproducible from its
inputs, no wheel, no model download and no native build — which is what lets
the whole of this milestone be tested without a microphone.

### Endpoint detection

Voice activity answers "is there speech in this frame?"; endpointing answers
"has the caller stopped?", which is harder, because speech is full of gaps.

```
IDLE ──speech ≥ 120 ms──► SPEAKING ──silence ≥ 700 ms──► ended
                          SPEAKING ──length ≥ 20 s────► ended (truncated)
```

A **pre-roll buffer** keeps the 200 ms before the detector was convinced, so
the first syllable survives — milestone 6 discarded it and clipped the start
of every sentence. A **minimum** means a click or a cough never reaches a
model. A **ceiling** answers a caller who never pauses, and bounds the buffer
on an open socket. A caller who says nothing produces no utterance, no
recognition and no model call at all.

### Barge-in, and what cancellation can and cannot do

When speech is detected while the receptionist is talking: the generation is
bumped, the turn is cancelled, the synthesis stream is closed, the outbound
queue is drained, and the transport is told to discard what it has buffered —
`{"event":"clear"}` for the carrier, `{"type":"clear"}` for the browser.

**Every unit of work carries a generation number**, and it is checked in four
places: before a final transcript reaches the dialogue, before synthesis
starts, **before every chunk of audio is written**, and after a cancelled task
returns anyway. The third is the one that matters — a provider can have chunks
in flight when it is closed.

**Cancellation is partial, by nature.** `anyio.to_thread.run_sync` cannot
terminate a running thread, so a model request and the tool calls it already
made will finish after an interruption. That means:

* **A booking that was already committed stays committed.** Undoing a real
  appointment because the caller started talking would be worse than letting
  it stand. There is a test that asserts the row survives.
* **Its reply is discarded and never heard.**
* **The dialogue is never run twice**, so nothing is done twice.
* **The transcript of an interrupted turn still stands** — it is a record of
  what happened, not of what was heard.

This is deliberate, and it is the honest limit of interrupting work that has
already reached a database.

### Streaming providers

Two new vendor-neutral interfaces, beside the batch ones rather than replacing
them: `StreamingSpeechToText` (`stream → send → finish → partials, one final`)
and `StreamingTextToSpeech` (`stream → chunks`). Chunks carry headerless PCM,
because a WAV header per chunk is a container the caller cannot chain.

**A partial transcript never reaches the dialogue layer.** Partials are shown
and logged; only a final crosses into `Conversation`, because a tool called
from a guess is a booking made from a guess. Two tests exist purely to prove
it — one at the session, one through the browser socket.

**This streams audio, not model tokens.** The dialogue still produces one
complete reply; the tool loop and the prompt are untouched. What is streamed
is the synthesis of that finished sentence, which is where the waiting is.

`offline_streaming.py` is the default and what every test uses: scripted
partials and a tone, no network, no key. `deepgram_stream_stt.py` and
`elevenlabs_stream_tts.py` speak their services' WebSocket protocols over the
`websockets` client already present — no vendor SDK, and each is the only
module in the project that knows its service's wire format.

**Both were written from documented protocol knowledge and have never been run
against the real services from this repository.** Their frame shapes are
verified against a fake connection; recognition quality, voice quality,
latency and reconnect behaviour are not claimed and have not been observed. If
a field name disagrees with a vendor's current documentation, believe the
documentation.

On a mid-utterance provider disconnect there is **no reconnect-and-resume**: a
reconnect loses the audio the service had buffered, and a silently truncated
transcript is worse than asking the caller to repeat themselves.

### Latency

Eight timestamps are captured per turn, and four numbers derived from them:

| Number | Definition |
|---|---|
| `audio_ms` | `speech_end − speech_start` — how long the caller spoke |
| `stt_latency_ms` | `stt_final − speech_end` — recognition after they stopped |
| `tts_latency_ms` | `tts_first_audio − dialogue_end` — time to the first audio |
| `first_audio_latency_ms` | `tts_first_audio − speech_end` — **the specification's number** |
| `turn_latency_ms` | `playback_end − speech_end` |

The first three are written to `turns.audio_ms`, `turns.stt_latency_ms` and
`turns.tts_latency_ms` — the columns that have been null since milestone 1.
`llm_latency_ms` remains milestone 4's and is not duplicated. `playback_end`
is held in session state and stored nowhere: a column for a number the carrier
may never report would be worse than deriving one.

This required the smallest possible change to the dialogue layer:
`DialogueResult` now also carries the two turn ids it wrote. Nothing in that
package reads them; they exist so a layer that can measure what the dialogue
cannot has somewhere to put it. No behaviour changed, and **no migration was
added — there are still two.**

### p50 / p95

```bash
python -m app.metrics           # or --call <id> for one call
```

```
metric                         count       p50       p95
--------------------------------------------------------
first_audio_latency_ms            34       910      1420
caller_audio_ms                   34      1800      3100
```

Percentiles come from `statistics.quantiles`. Rows from before realtime have
null latencies and are **skipped rather than counted as zero**; a turn missing
any one stage is not measured at all, because a partial sum would read as a
fast turn. Below twenty samples the table marks the row and says the numbers
are arithmetic rather than evidence. No endpoint, no dashboard, no time series.

**It measures; it does not promise.** Whether the specification's 1.2-second
target is met depends on provider round-trips this repository has never made.

### What has and has not actually been proved

The tests and the simulated smoke runs drive **real** WebSockets against the
**real** application and a **real** PostgreSQL database, with scripted model
and speech providers and a simulated carrier. They exercise voice activity
detection, endpointing, the streaming interfaces, the generation model,
barge-in, cancellation, the dialogue path, latency capture and the call
lifecycle.

**They do not place a telephone call and they do not call any external
service.** There is no Anthropic, Deepgram, ElevenLabs or Twilio credential in
this development environment, and none has ever been used. Nothing here
establishes real carrier behaviour, real latency, real transcription accuracy,
real voice quality or production readiness, and no such claim is made anywhere
in this repository.

### What milestone 8 built

Cost tracking: what each call consumed, recorded per component, and priced
only where somebody has supplied a price.

```
turn finishes ─► app/cost/recorder.py ─► call_costs (llm, stt, tts)
call ends     ─► app/cost/recorder.py ─► call_costs (telephony)
                                              │
                                              └─► calls.total_cost_usd
```

* `app/models/call_cost.py` — the `call_costs` table: one row per component
  per turn, plus one per call for the line. Usage in named units, an optional
  per-unit price, an optional cost, and a JSONB `metadata` column recording
  what was measured and why anything is unpriced.
* `app/cost/usage.py` — vendor-neutral usage records. Tokens, milliseconds of
  audio, characters, milliseconds of line time. No prices anywhere in it.
* `app/cost/pricing.py` — the only module that knows a price.
* `app/cost/recorder.py` — the only module that writes a cost row or touches
  `calls.total_cost_usd`.
* One migration, `11bc87a6af66`. The two existing migrations are untouched.

Token counts now travel out of the dialogue layer on `DialogueResult`
(`model_name`, `input_tokens`, `output_tokens`), summed across every model
request one caller turn made — a turn that went round the tool loop four times
paid for four requests. Both session objects carry the speech usage they
measured (`stt_provider`, `stt_audio_ms`, `tts_provider`, `tts_characters`).
Neither layer knows what any of it costs; the transports hand the numbers to
the cost recorder, beside the latency writer from milestone 7.

### VoiceDesk ships no prices

This is the milestone's central decision and it is not a limitation to be
fixed later without thought.

**The built-in pricing table contains no vendor price and never should.** Not
Anthropic's, not Deepgram's, not ElevenLabs', not Twilio's. Prices change,
this repository cannot check them, and a stale number baked into source is
worse than no number: it would be quoted as a fact for as long as it survived.

So a price comes from configuration or from nowhere:

```bash
export VOICEDESK_COST_TRACKING_ENABLED=true
export VOICEDESK_LLM_INPUT_USD_PER_MTOK=...    # from your own account
export VOICEDESK_LLM_OUTPUT_USD_PER_MTOK=...
export VOICEDESK_STT_USD_PER_MINUTE=...
export VOICEDESK_TTS_USD_PER_MCHAR=...
```

Blank means **unpriced**, which is not the same as free. An unpriced component
is stored with its usage and a null `cost_usd`, and a call with any unpriced
component has a null `total_cost_usd` — a partial total looks exactly like a
complete one and would be read as a complete one.

The one thing the shipped table does contain is the `offline` providers, at
exactly zero. That is not a guess about anybody's price list: the offline
transcriber and synthesiser are a fixed sentence and a tone written in this
repository, and they buy nothing. Configuration cannot override that.

Malformed configuration fails at startup rather than one swallowed exception
at a time: `create_app()` parses the prices when cost tracking is on, and a
price that is not a number, is negative, or is large enough to only be a
misplaced decimal point stops the process.

### What these numbers are and are not

They are what VoiceDesk measured, multiplied by what an operator said things
cost. **They are not a bill, and they will not match one.** Specifically:

* **The line is never priced.** What this code can measure is the wall-clock
  window in which a media stream was open. What a carrier bills is its own
  record of the call, rounded up to whole units, which this process never
  sees. The duration is recorded with `unit_type = "duration_ms"` and no rate,
  so as it stands **a telephone call has no `total_cost_usd` at all**. That is
  the honest consequence of not pricing something we cannot measure, and it is
  asserted in the tests rather than left as a surprise.
* **Cache tokens are not measured.** The provider response this repository
  reads does not expose cache reads or writes, so a cached prompt is priced as
  though it were not cached — an over-count, in the direction of a larger bill
  than the real one.
* **Streaming recognition under-counts.** `audio_ms` comes from the transcript
  results, not from how long the connection was open, and a streaming
  recogniser is usually billed for the latter.
* **A greeting is not accounted for.** It is synthesised before anybody has
  spoken, so there is no turn row to attach it to, and inventing one would put
  a sentence in the transcript that nobody said.
* **A turn the model never saw records nothing.** When a caller says nothing
  recognisable the model is deliberately not asked, so no turn rows are
  written — and a cost row hangs off a turn. The recognition and synthesis
  that did happen are not recorded.
* **Sub-microdollar amounts round to zero.** `cost_usd` is `Numeric(10, 6)`.

The specification asks for "cost per call and cost per completed booking" in
this README as the sales argument. **No such figure is given here, because
none has been earned:** no external service has ever been called from this
code, so there is no usage from a real call to price, and no price to apply to
it. The machinery to produce that number exists and is tested; the number
does not.

### Counting characters once

`SpeechChunk.characters` is a trap worth naming. Both streaming synthesisers
in this repository — ElevenLabs and offline — report the **whole** text length
on **every** chunk, and so do the fakes in the tests. A layer that
summed them would multiply the bill by the number of chunks: a hundred
characters delivered in ten chunks would be recorded as a thousand.

So the realtime session counts once, from the text it handed to the
synthesiser, before the first chunk comes back. `tests/test_cost_integration.py`
and `tests/test_realtime_session.py` both drive a ten-chunk stream of a
hundred-character reply and assert the recorded figure is 100.

### Recording twice records once

Cost rows are keyed by `(turn_id, component)` with a unique index, so a
retried write updates the row it already made rather than doubling a call's
spend. PostgreSQL treats nulls in a unique index as distinct, which would let
a second telephony row through, so a **partial** unique index on
`(call_id, component) WHERE turn_id IS NULL` covers exactly the rows the first
one cannot reach. Both are asserted directly against the database.

`calls.total_cost_usd` is recomputed with a single aggregate over
`call_costs` — never by adding to whatever the column happened to hold.
Read-modify-write on a column two turns can finish at the same time is how
totals drift.

### Decisions recorded in milestone 8

| # | Decision | Why |
|---|---|---|
| D1 | Configuration-only pricing; the shipped table holds no vendor price | This environment cannot verify any vendor's current prices, and a stale one in source would be quoted as a fact |
| D2 | Blank price means unpriced, not zero; a numeric setting would default to 0.0 | Pricing every call at nothing by accident is the one failure nobody would notice |
| D3 | One unpriced component nulls the whole call's total | A partial total is indistinguishable from a complete one |
| D4 | The offline providers are priced at exactly zero, and configuration cannot change it | They are this repository's own code and buy nothing; that is a fact, not a price |
| D5 | Telephony is recorded unpriced, always | Measured stream time is not the carrier's billed time, and pricing it would produce a number that looked like a bill |
| D6 | `unit_price_usd` is null for a model row, with both rates in `metadata` | One column cannot honestly hold two rates |
| D7 | Rates are stored per single unit, quantised before use | So `cost_usd = unit_price_usd × units` is checkable in SQL, not only in Python |
| D8 | Recognition is recorded against the caller's turn; the model and synthesis against the reply | The same split the milestone-7 latency writer already uses |
| D9 | Cost tracking is off by default | A fresh clone behaves exactly as it did through milestone 7 |
| D10 | Prices are validated at application start, not at first use | A malformed price should stop a deployment, not quietly disable accounting |
| D11 | `Decimal` and `Numeric` throughout; floats are refused by `Usage` | Money computed from a float is wrong eventually and silently |
| D12 | One new migration; the existing two are untouched | A new table, altering nothing, is safe against a populated database |

### Deliberately not built yet

No outbound calling, SMS, email, actual transfer, scenario evals or
deployment. **No billing:** there is no Stripe, no invoice, no subscription,
no quota, no customer account and no payment processing, and none is coming.
Milestone 8 is accounting — writing down what was used — and nothing else.

**No external service has ever been called from this code, and no telephone
call has ever been placed.** Every test injects a scripted model and scripted
speech providers, and the telephony tests simulate the carrier over a real
socket. The Anthropic, Deepgram, ElevenLabs and Twilio request shapes are
verified; none of their behaviour is. No accuracy, latency, quality or cost
number is claimed for any of them; that starts with the milestone-9 scenario
evals.

`transfer_to_human` records the intent to escalate and the reason. Actually
connecting the caller to a person is not implemented.

There is **no dialogue API and no booking API**: the calendar core, the tool
layer and the dialogue layer are libraries. The HTTP surface is `GET /health`,
`GET /harness` and `POST /telephony/voice`, plus two sockets — `WS /ws/harness`
and `WS /telephony/stream`.

`AppointmentStatus` has two values, `booked` and `cancelled`: `reschedule`
moves an existing booking's times and leaves it `booked`, so it is not a third
state. `calls.direction` allows `outbound` because a direction has two values,
but v1 only ever writes `inbound` — outbound calling is a non-goal.

### Running locally

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

**The browser harness.** With the server running, open
<http://localhost:8000/harness>, press Connect, then press and hold *Hold to
talk* and release to send. You will need at least one active row in `services`
and some `business_hours` for the calendar to have anything to offer.

Out of the box the speech providers are offline, so you can do that with no
credentials at all — but remember what you are hearing: whatever you say, the
transcriber returns the same fixed sentence, and the reply is a tone rather
than a voice. For real speech:

```bash
export VOICEDESK_STT_PROVIDER=deepgram VOICEDESK_DEEPGRAM_API_KEY=...
export VOICEDESK_TTS_PROVIDER=elevenlabs VOICEDESK_ELEVENLABS_API_KEY=...
export VOICEDESK_TTS_VOICE=...            # an ElevenLabs voice id
```

The dialogue itself needs `VOICEDESK_ANTHROPIC_API_KEY` whichever speech
providers you use. Without it the harness still connects and plays its
greeting, and each turn comes back as a `turn_failed` message rather than
dropping the socket.

Alembic reads the database URL from `VOICEDESK_DATABASE_URL` via
`app/config.py`; `alembic.ini` deliberately holds no URL, so migrations and the
app cannot disagree about which database they are using. Tests that need
PostgreSQL are skipped when no server answers, so `pytest` still runs without
one (576 pass, 495 skip). With a database: 1071 pass.

VoiceDesk is a separate application from DocIntel in this repository: its own
package, dependencies, virtualenv, configuration prefix and database. Nothing
is shared between them.
