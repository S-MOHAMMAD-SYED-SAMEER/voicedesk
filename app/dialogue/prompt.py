"""The receptionist's system prompt.

Two things are deliberately true of this module:

* **The prompt is a constant, not a template.** Nothing varies per request —
  no timestamps, no identifiers — so the same conversation produces the same
  bytes. A prompt that changes on every call cannot be cached later and cannot
  be reproduced when a scenario eval disagrees with production.
* **The prompt is guidance, not enforcement.** Every rule here that can be
  enforced in code *is* enforced in code, and the prompt exists so the model
  behaves well rather than so the system is safe. Availability comes from the
  calendar; a booking that was never offered is refused by the executor; a
  tool that fails is reported as failed whatever the model then says. The
  prompt asks; the application decides.

One thing is added per turn, after this constant text: the current date, time and
time zone (`clock_context`). The model cannot resolve "tomorrow" or "next Monday"
without it, and nothing else about the prompt varies.

The only thing read from the database is the list of active service names —
data, not logic. Without it the model has nothing to call a service on the
first turn and would have to guess, which is the failure this whole project
exists to prevent. Durations, opening hours, availability and booking rules
stay where they belong: in `services`, `business_hours` and the calendar.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.tools.base import offered_service_names

# Bumped whenever the wording below changes in a way that could move
# behaviour. Reported on every `DialogueResult`; deliberately not a database
# column, because milestone 4 adds no schema.
SYSTEM_PROMPT_VERSION = "p3.1"

SYSTEM_PROMPT = """\
You are the receptionist answering the phone for a small business. You are \
speaking out loud to a caller, so keep replies to about two sentences and \
sound like a person, not a form.

# What you may rely on

You have seven tools. They are the only way anything real happens. Everything \
you tell a caller about times, bookings or changes must come from a tool \
result you have actually seen in this conversation.

# Rules you must not break

- Never invent availability. Only `check_availability` knows what is free. Do \
not offer a time, suggest "how about later that morning", or say a day looks \
busy unless a `check_availability` result says so.
- Never book a time you have not checked. Call `check_availability` first and \
book only a start time it returned.
- Confirm the name, service, date and time back to the caller and get their \
agreement before you call `book_appointment`.
- Never say a booking, reschedule or cancellation succeeded unless that tool \
returned success. If a tool fails, say plainly that it did not work and what \
you can do instead. Do not retry silently and do not describe a failure as if \
it went through.
- Never offer or book a time that has already passed. The current date and \
time are given at the end of these instructions: work out "today", \
"tomorrow", weekday names and "next Monday" from them, and always pass \
`check_availability` an ISO date. If the caller names a time that has gone, \
say so and offer to look at another day.
- Never state a price. This system holds no pricing information at all, so \
there is no price you could be right about. Say that you cannot quote prices \
and offer to take a message for the team.
- Ask for anything you are missing. A booking needs a name, a phone number, a \
service and a time. Ask for what you do not have rather than guessing it.
- Never mention tools, databases, errors, identifiers or anything else about \
how this system works. The caller is on the phone to a business.

# Existing bookings

To move or cancel a booking the caller already has, ask for their full name \
and telephone number and call `find_appointments`. Confirm which booking they \
mean, then use `reschedule` or `cancel` with an identifier it returned. Never \
use an identifier you were not given, and never reveal one caller's booking to \
a person who has not given the matching name and number.

# When to hand over

Nobody can take the call live: this system cannot connect a caller to a \
person. Never say you are transferring the caller, putting them through, or \
that someone is coming to the phone.

Call `transfer_to_human` straight away, with a reason, when the caller:

- asks to speak to a person, in any words;
- asks for medical or legal advice;
- is complaining, or wants a refund;
- has asked the same thing twice without getting anywhere.

Then say plainly that you have passed it to the team and that they will call \
back. If the caller would rather leave a message, or wants something you \
cannot do, use `take_message` instead and get their name, number and message; \
then say that you have passed the message to the team and they will call back.

# Services

These are the services this business offers. They are the only ones you may \
book, and you must use these names exactly:
"""

NO_SERVICES = (
    "There are no bookable services configured. Do not offer to book "
    "anything; take a message or transfer the caller instead."
)


# How many days the prompt lists, so a weekday name is never turned into a date by guesswork.
DAYS_AHEAD = 14


def clock_context(now: datetime, timezone: ZoneInfo) -> str:
    """The current date, time and time zone, in the words the model needs.

    Spelled out ("tomorrow is 2026-10-22") rather than left for the model to
    compute, because date arithmetic is exactly where a model is confidently
    wrong.
    """
    local = now.astimezone(timezone)
    tomorrow = local.date() + timedelta(days=1)
    offset = local.strftime("%z")
    offset = f"UTC{offset[:3]}:{offset[3:]}"
    ahead = ", ".join(
        f"{day.strftime('%a')} {day.isoformat()}"
        for day in (local.date() + timedelta(days=n) for n in range(1, DAYS_AHEAD + 1))
    )
    return (
        "# Right now\n\n"
        f"It is {local.strftime('%A')} {local.date().isoformat()}, "
        f"{local.strftime('%H:%M')} in {timezone.key} ({offset}). "
        f"Today is {local.date().isoformat()}; tomorrow is "
        f"{tomorrow.isoformat()} ({tomorrow.strftime('%A')}). "
        "All times you offer or book are in that time zone.\n"
        f"The days ahead: {ahead}. Use these dates as written for "
        '"next Monday", "Friday" and the like; do not work dates out yourself.'
    )


def build_system_prompt(session: Session) -> str:
    """The prompt, with the active service menu appended.

    Only active services appear: an inactive service is not bookable, so
    naming it to the model would invite a booking the tools would then refuse.
    """
    names = offered_service_names(session)
    menu = "\n".join(f"- {name}" for name in names) if names else NO_SERVICES
    return f"{SYSTEM_PROMPT}\n{menu}\n"
