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
