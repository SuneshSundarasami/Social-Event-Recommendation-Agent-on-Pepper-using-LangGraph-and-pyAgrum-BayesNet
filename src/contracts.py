"""Shared contracts for the Social Event Recommendation Agent.

This module is the single source of truth for the data types and interfaces that
flow between work packages (WP1 perception/FSM, WP2 dialogue, WP3 recommender,
WP4 behaviour). Freezing these lets each package be built and tested in
isolation, then snapped together.

Kept deliberately dependency-free and Python 3.8 compatible so it can be
imported by both the robot side (3.8 / NAOqi / qiBullet) and the dialogue
service (3.10+ / LangGraph).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

# ---------------------------------------------------------------------------
# Evidence (the single currency passed between packages)
# ---------------------------------------------------------------------------

# Legal label values for each evidence slot. Any slot may be MISSING from a
# concrete Evidence dict — partial evidence is explicitly allowed (WP3 runs the
# Bayesian network with whatever is available).
EVIDENCE_LABELS: Dict[str, Tuple[str, ...]] = {
    "Budget":        ("Low", "Med", "High"),
    "GroupSize":     ("Solo", "Small", "Large"),
    "ActivityLevel": ("Relaxed", "Moderate", "Active"),
    "Setting":       ("Indoor", "Outdoor", "Either"),
    "TimeOfDay":     ("Day", "Evening", "Night"),
    "Interest":      ("Arts", "Music", "Food", "Sports"),
}

# Canonical question order (also the default slot-asking priority for WP2).
EVIDENCE_SLOTS: Tuple[str, ...] = tuple(EVIDENCE_LABELS.keys())

# The eight event categories the recommender ranks over (output layer).
EVENT_LABELS: Tuple[str, ...] = (
    "Museum",
    "Concert",
    "Sports",
    "Food",
    "Outdoor",
    "Nightlife",
    "Workshop",
    "Networking",
)

# Type aliases (documentation; not enforced at runtime).
Evidence = Dict[str, str]                 # e.g. {"Budget": "Low", "Setting": "Outdoor"}
Result = List[Dict[str, object]]          # [{"event": "Museum", "prob": 0.31}, ...] sorted desc


def is_valid_evidence_value(slot: str, value: str) -> bool:
    """True if ``value`` is a legal label for ``slot``."""
    return slot in EVIDENCE_LABELS and value in EVIDENCE_LABELS[slot]


def validate_evidence(evidence: Evidence) -> Evidence:
    """Return a clean copy of ``evidence`` containing only legal slot/value pairs.

    Unknown slots and illegal values are dropped (defensive guard so an LLM can
    never inject values the Bayesian network does not understand).
    """
    return {
        slot: value
        for slot, value in (evidence or {}).items()
        if is_valid_evidence_value(slot, value)
    }


def missing_slots(evidence: Evidence) -> List[str]:
    """Slots (in canonical order) not yet filled with a legal value."""
    clean = validate_evidence(evidence)
    return [slot for slot in EVIDENCE_SLOTS if slot not in clean]


# ---------------------------------------------------------------------------
# Cross-package interfaces (Protocols — structural typing, no inheritance needed)
# ---------------------------------------------------------------------------

@runtime_checkable
class Perception(Protocol):
    """WP1 perception source consumed by the state machine."""

    def tick(self) -> bool:
        """Grab/process one frame; return whether a face was detected this frame."""
        ...

    def face_stable(self) -> bool:
        """True once a face has been detected for enough recent frames (Idle->Greeting)."""
        ...

    def is_present(self) -> bool:
        """True if the user appears to still be present (looser than ``face_stable``)."""
        ...

    def reset(self) -> None:
        """Clear the smoothing window (e.g. when returning to Idle)."""
        ...

    def release(self) -> None:
        """Release any underlying camera resources."""
        ...


@runtime_checkable
class DialogueManager(Protocol):
    """WP2 — collects user preferences and returns validated evidence."""

    def collect_evidence(self) -> Evidence:
        ...


@runtime_checkable
class Recommender(Protocol):
    """WP3 — Bayesian recommendation engine."""

    def recommend(self, evidence: Evidence) -> Result:
        ...

    def explain(self, event: str, evidence: Evidence) -> str:
        ...


@runtime_checkable
class Behaviour(Protocol):
    """WP4 — speech + gesture presentation layer."""

    def say(self, text: str) -> None:
        ...

    def gesture(self, name: str) -> None:
        ...

    def say_with_gesture(self, text: str, name: str) -> None:
        """Start speech and gesture simultaneously; block until BOTH finish."""
        ...

    def present(self, result: Result) -> None:
        ...
