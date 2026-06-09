"""DialogueManager — the WP2 entry point exposed to the rest of the system.

Implements the contract the robot side calls: ``collect_evidence() -> Evidence``.
"""

from __future__ import annotations

from dialogue.graph import build_graph, initial_state
from dialogue.schema import Evidence
from dialogue.slots import DEFAULT_SLOTS


class DialogueManager:
    def __init__(self, parser, input_provider, framer=None, slots=None,
                 max_clarify: int = 2, recursion_limit: int = 100) -> None:
        self.app = build_graph(parser, input_provider, framer,
                               slots or DEFAULT_SLOTS, max_clarify)
        self._config = {"recursion_limit": recursion_limit}

    def collect_evidence(self) -> Evidence:
        final = self.app.invoke(initial_state(), self._config)
        return final["evidence"]

    def run_verbose(self) -> dict:
        """Run and return the full final state (evidence + transcript)."""
        return self.app.invoke(initial_state(), self._config)
