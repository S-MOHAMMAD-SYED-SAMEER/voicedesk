"""The evaluation suite: does the receptionist behave, measured deterministically.

    scenario → world → ScriptedModel → Conversation → tools → calendar → PostgreSQL
                                                                   ↓
                                              trace → checks → findings → report

Only the model is scripted. The dialogue layer, the executor and its
offered-slot guard, all six tools, `CalendarService` and the database's
exclusion constraint are the ones the application runs, against an isolated
`_evals` database that this package creates, migrates and truncates.

**What this measures.** VoiceDesk's behaviour: whether the right tool was
called with the right arguments, whether a booking that succeeded was for a
time the calendar had actually offered, whether an escalation happened when
the scenario said it should, and what the database held when the call ended.

**What this does not measure.** Any real model's conversational quality. The
model here says what the scenario told it to say; it has no judgement to
assess, and no number produced by this suite is evidence about Claude, about
Deepgram, about ElevenLabs or about a carrier. Nothing in this package calls
any of them, and no credential is needed to run it.

There is no LLM-as-judge here and there should never be one. Every verdict is
a comparison of values the system produced against values a scenario declared.
"""

# Bumped when a definition or a ground-truth semantic changes in a way that
# makes two runs incomparable — a new blocking category, a changed task-success
# rule. Adding a scenario does not need a bump; redefining what passing means
# does. Reported at the top of every run, so a number can always be traced to
# the rules that produced it.
EVAL_SUITE_VERSION = "m9-v1"

__all__ = ["EVAL_SUITE_VERSION"]
