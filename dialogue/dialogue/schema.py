"""Evidence schema for the dialogue service.

MUST stay in sync with ``Project/src/contracts.py`` (EVIDENCE_LABELS) — the
contract between WP2 and the rest of the system is the JSON shape of this dict,
so the two are intentionally duplicated across the 3.8 and 3.11 projects.
"""

from __future__ import annotations

# Legal label values for each evidence slot.
EVIDENCE_LABELS: dict[str, tuple[str, ...]] = {
    "Budget":        ("Low", "Med", "High"),
    "GroupSize":     ("Solo", "Small", "Large"),
    "ActivityLevel": ("Relaxed", "Moderate", "Active"),
    "Setting":       ("Indoor", "Outdoor", "Either"),
    "TimeOfDay":     ("Day", "Evening", "Night"),
    "Interest":      ("Arts", "Music", "Food", "Sports"),
}

EVIDENCE_SLOTS: tuple[str, ...] = tuple(EVIDENCE_LABELS.keys())

Evidence = dict[str, str]


def is_valid(slot: str, value: str) -> bool:
    return slot in EVIDENCE_LABELS and value in EVIDENCE_LABELS[slot]


def whitelist(candidate: dict) -> Evidence:
    """Keep only legal slot/value pairs (defensive guard against bad LLM output)."""
    return {k: v for k, v in (candidate or {}).items() if is_valid(k, v)}
