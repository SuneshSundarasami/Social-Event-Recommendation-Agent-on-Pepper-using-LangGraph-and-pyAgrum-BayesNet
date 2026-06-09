"""Offline graph tests — scripted parser + scripted input, no network/mic."""

from dialogue.inputs import ScriptedInput
from dialogue.llm import ScriptedParser
from dialogue.manager import DialogueManager
from dialogue.schema import EVIDENCE_SLOTS, is_valid


def test_collects_all_slots_from_clear_answers():
    answers = [
        "something cheap",
        "with a few friends",
        "fairly relaxed",
        "outdoors",
        "during the day",
        "i love food",
    ]
    mgr = DialogueManager(ScriptedParser(), ScriptedInput(answers))
    ev = mgr.collect_evidence()
    assert ev == {
        "Budget": "Low",
        "GroupSize": "Small",
        "ActivityLevel": "Relaxed",
        "Setting": "Outdoor",
        "TimeOfDay": "Day",
        "Interest": "Food",
    }


def test_multi_slot_fill_from_one_answer():
    # One rich answer fills several slots; the rest are asked normally.
    answers = [
        "cheap, outdoors, with a few friends",  # Budget + Setting + GroupSize
        "relaxed",
        "daytime",
        "food",
    ]
    mgr = DialogueManager(ScriptedParser(), ScriptedInput(answers))
    ev = mgr.collect_evidence()
    assert ev["Budget"] == "Low"
    assert ev["Setting"] == "Outdoor"
    assert ev["GroupSize"] == "Small"
    assert set(ev) == set(EVIDENCE_SLOTS)


def test_default_applied_after_failed_clarifications():
    # Gibberish for Budget -> after retries a default is used; later answers fine.
    answers = [
        "asdfgh", "qwerty", "zzzzz",          # Budget never parses -> default Med
        "solo", "active", "indoor", "night", "music",
    ]
    mgr = DialogueManager(ScriptedParser(), ScriptedInput(answers), max_clarify=2)
    ev = mgr.collect_evidence()
    assert ev["Budget"] == "Med"            # default
    assert set(ev) == set(EVIDENCE_SLOTS)
    assert all(is_valid(s, v) for s, v in ev.items())
