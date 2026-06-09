"""Slot configuration - the data-driven question plan (PLAN.md §5).

A single generic slot-filling loop reads this table, so adding/reordering
questions is just editing this list.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialogue.schema import EVIDENCE_LABELS


@dataclass(frozen=True)
class Slot:
    name: str
    question: str       # asked when the slot is empty
    clarify: str        # re-asked when the answer was ambiguous
    default: str        # used once clarification attempts are exhausted

    @property
    def options(self) -> tuple[str, ...]:
        return EVIDENCE_LABELS[self.name]


DEFAULT_SLOTS: list[Slot] = [
    Slot("Budget",
         "What's your budget like - low, medium, or high?",
         "Sorry, would you say your budget is low, medium, or high?",
         "Med"),
    Slot("GroupSize",
         "Are you going solo, with a small group, or a large group?",
         "Just to check - solo, a small group, or a large group?",
         "Small"),
    Slot("ActivityLevel",
         "Do you feel like something relaxed, moderate, or active?",
         "Would you prefer relaxed, moderate, or active?",
         "Moderate"),
    Slot("Setting",
         "Indoor, outdoor, or no preference?",
         "Should it be indoor, outdoor, or either is fine?",
         "Either"),
    Slot("TimeOfDay",
         "When are you thinking - daytime, evening, or night?",
         "Daytime, evening, or night?",
         "Evening"),
    Slot("Interest",
         "What are you into - arts, music, food, or sports?",
         "Which one sounds best: arts, music, food, or sports?",
         "Music"),
]


# Short natural-language hints, given to the LLM question framer so it asks about
# each unknown field meaningfully (without reading out the option list).
SLOT_HINTS: dict[str, str] = {
    "Budget": "how much they'd like to spend (cheap / mid-range / pricey)",
    "GroupSize": "whether they're coming alone, with a few people, or a big group",
    "ActivityLevel": "how relaxed or active they want it to be",
    "Setting": "whether they prefer indoors or outdoors",
    "TimeOfDay": "what time of day suits them (daytime / evening / night)",
    "Interest": "what they enjoy (arts, music, food, or sports)",
}
