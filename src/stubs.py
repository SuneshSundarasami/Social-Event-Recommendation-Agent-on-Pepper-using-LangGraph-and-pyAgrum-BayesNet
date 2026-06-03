"""Console / scripted stubs for WP2, WP3, WP4.

These let WP1 (perception + FSM) run end-to-end today. Each is a drop-in that
satisfies the matching ``contracts`` interface and will be replaced by the real
work package later:

  * ConsoleBehaviour     -> WP4 (prints speech/gestures to the terminal)
  * StubDialogueManager  -> WP2 (scripted or interactive console answers)
  * StubRecommender      -> WP3 (simple deterministic ranking, no Bayes yet)
"""

from __future__ import annotations

from typing import Optional

from contracts import (
    EVENT_LABELS,
    EVIDENCE_LABELS,
    EVIDENCE_SLOTS,
    Evidence,
    Result,
    is_valid_evidence_value,
)


# ---------------------------------------------------------------------------
# WP4 stub
# ---------------------------------------------------------------------------

class ConsoleBehaviour:
    """Prints what Pepper would say/do. Replaced by the qiBullet behaviour layer."""

    def say(self, text: str) -> None:
        print("  [Pepper] (says)    %s" % text)

    def gesture(self, name: str) -> None:
        print("  [Pepper] (gesture) %s" % name)

    def present(self, result: Result) -> None:
        print("  [Pepper] (presents top events):")
        for i, item in enumerate(result[:3], start=1):
            event = item.get("event", "?")
            prob = item.get("prob", 0.0)
            try:
                prob_str = "%.0f%%" % (float(prob) * 100)
            except Exception:
                prob_str = str(prob)
            print("      %d. %-12s %s" % (i, event, prob_str))


# ---------------------------------------------------------------------------
# WP2 stub
# ---------------------------------------------------------------------------

class StubDialogueManager:
    """Returns evidence either from a fixed dict or by asking on the console.

    interactive=False -> returns ``scripted`` (or a sensible default).
    interactive=True  -> asks one multiple-choice question per slot.
    """

    def __init__(self, scripted: Optional[Evidence] = None, interactive: bool = False) -> None:
        self.scripted = scripted
        self.interactive = interactive

    def collect_evidence(self) -> Evidence:
        if not self.interactive:
            return dict(self.scripted) if self.scripted else self._default()

        evidence: Evidence = {}
        print("  [Dialogue] Please answer a few questions (Enter to skip):")
        for slot in EVIDENCE_SLOTS:
            options = EVIDENCE_LABELS[slot]
            prompt = "    %s %s: " % (slot, list(options))
            try:
                answer = input(prompt).strip()
            except EOFError:
                answer = ""
            if answer and is_valid_evidence_value(slot, answer):
                evidence[slot] = answer
        return evidence

    @staticmethod
    def _default() -> Evidence:
        return {
            "Budget": "Low",
            "GroupSize": "Small",
            "ActivityLevel": "Relaxed",
            "Setting": "Outdoor",
            "TimeOfDay": "Day",
            "Interest": "Food",
        }


# ---------------------------------------------------------------------------
# WP3 stub
# ---------------------------------------------------------------------------

class StubRecommender:
    """Toy deterministic recommender so the FSM has something to present.

    Maps the Interest slot to a couple of likely events and returns a small
    ranked list. This is a placeholder for the real pgmpy/pyAgrum network (WP3).
    """

    _INTEREST_BIAS = {
        "Arts": ["Museum", "Workshop"],
        "Music": ["Concert", "Nightlife"],
        "Food": ["Food", "Networking"],
        "Sports": ["Sports", "Outdoor"],
    }

    def recommend(self, evidence: Evidence) -> Result:
        interest = evidence.get("Interest")
        preferred = self._INTEREST_BIAS.get(interest, [])
        if evidence.get("Setting") == "Outdoor" and "Outdoor" not in preferred:
            preferred = preferred + ["Outdoor"]

        ordered = preferred + [e for e in EVENT_LABELS if e not in preferred]
        n = len(ordered)
        # Linearly decreasing pseudo-probabilities that sum to 1.
        weights = [n - i for i in range(n)]
        total = float(sum(weights))
        return [
            {"event": event, "prob": weights[i] / total}
            for i, event in enumerate(ordered)
        ]

    def explain(self, event: str, evidence: Evidence) -> str:
        bits = []
        if "Interest" in evidence:
            bits.append("you're interested in %s" % evidence["Interest"].lower())
        if "Setting" in evidence:
            bits.append("you prefer an %s setting" % evidence["Setting"].lower())
        if "Budget" in evidence:
            bits.append("a %s budget" % evidence["Budget"].lower())
        reason = ", ".join(bits) if bits else "your preferences"
        return "I suggest %s because %s." % (event, reason)
