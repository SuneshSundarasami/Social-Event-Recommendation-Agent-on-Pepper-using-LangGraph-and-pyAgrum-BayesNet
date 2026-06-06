"""Six-state interaction FSM (WP1) — the orchestration spine.

Idle -> Greeting -> Conversation -> Reasoning -> Recommendation -> Farewell -> Idle

The FSM owns the control loop and calls into the other work packages purely
through the ``contracts`` interfaces, so any of them can be a stub:
  * Perception     (WP1)  -- face detection / presence
  * DialogueManager(WP2)  -- collect_evidence()
  * Recommender    (WP3)  -- recommend() / explain()
  * Behaviour      (WP4)  -- say() / gesture() / present()

Robustness (proposal section 6) is handled here: if the user disappears during
the interaction the FSM resets to Idle, and a weak/empty recommendation still
presents the top events that are available.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Callable, List, Optional

from contracts import (
    Behaviour,
    DialogueManager,
    Evidence,
    Perception,
    Recommender,
    Result,
    validate_evidence,
)

log = logging.getLogger("wp1.fsm")


class State(enum.Enum):
    IDLE = "Idle"
    GREETING = "Greeting"
    CONVERSATION = "Conversation"
    REASONING = "Reasoning"
    RECOMMENDATION = "Recommendation"
    FAREWELL = "Farewell"


class InteractionFSM:
    def __init__(
        self,
        perception: Perception,
        dialogue: DialogueManager,
        recommender: Recommender,
        behaviour: Behaviour,
        idle_poll: float = 0.1,
        presence_check_frames: int = 3,
        ack_timeout: float = 15.0,
        name_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.perception = perception
        self.dialogue = dialogue
        self.recommender = recommender
        self.behaviour = behaviour

        self.idle_poll = idle_poll
        self.presence_check_frames = presence_check_frames
        self.ack_timeout = ack_timeout
        self.name_provider = name_provider

        self.state = State.IDLE
        self.evidence: Evidence = {}
        self.result: Result = []
        self._running = False

    # -- public control -----------------------------------------------------

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Run the FSM loop. Stops after ``max_cycles`` completed interactions
        (each Farewell counts as one), or forever if ``max_cycles`` is None."""
        self._running = True
        cycles = 0
        handlers = {
            State.IDLE: self._idle,
            State.GREETING: self._greeting,
            State.CONVERSATION: self._conversation,
            State.REASONING: self._reasoning,
            State.RECOMMENDATION: self._recommendation,
            State.FAREWELL: self._farewell,
        }
        try:
            while self._running:
                log.debug("State -> %s", self.state.value)
                nxt = handlers[self.state]()
                if self.state == State.FAREWELL and nxt == State.IDLE:
                    cycles += 1
                    if max_cycles is not None and cycles >= max_cycles:
                        log.info("Reached max_cycles=%d; stopping.", max_cycles)
                        break
                self.state = nxt
        except KeyboardInterrupt:
            log.info("Interrupted by user; shutting down.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    # -- helpers ------------------------------------------------------------

    def _still_present(self) -> bool:
        """Sample a few frames and report whether the user is still there."""
        for _ in range(self.presence_check_frames):
            self.perception.tick()
            time.sleep(self.idle_poll)
        return self.perception.is_present()

    def _reset_interaction(self) -> None:
        self.evidence = {}
        self.result = []
        self.perception.reset()

    # -- state handlers (each returns the next State) -----------------------

    def _idle(self) -> State:
        # Passive monitoring until a face is stable for several frames.
        while self._running:
            self.perception.tick()
            if self.perception.face_stable():
                log.info("Stable face detected -> Greeting")
                return State.GREETING
            time.sleep(self.idle_poll)
        return State.IDLE

    def _greeting(self) -> State:
        name = None
        if self.name_provider is not None:
            try:
                name = self.name_provider()
            except Exception:
                name = None
        hello = "Hello %s!" % name if name else "Hello there!"
        # Two coordinated beats: each line and its gesture start together and
        # the FSM waits for both to finish before the next.
        self.behaviour.say_with_gesture(hello, "wave")
        self.behaviour.say_with_gesture("Nice to see you.", "nod")
        self.behaviour.say_with_gesture(
            "I can recommend a social event for you.", "open_arms")

        if not self._still_present():
            log.info("User left during greeting -> Idle")
            return State.IDLE
        return State.CONVERSATION

    def _conversation(self) -> State:
        raw = self.dialogue.collect_evidence()
        self.evidence = validate_evidence(raw)
        log.info("Collected evidence: %s", self.evidence)

        if not self._still_present():
            log.info("User left during conversation -> Idle")
            return State.IDLE
        return State.REASONING

    def _reasoning(self) -> State:
        self.behaviour.say_with_gesture("Let me think about that.", "think")
        self.result = self.recommender.recommend(self.evidence) or []
        log.info("Recommendation result: %s", self.result)
        return State.RECOMMENDATION

    def _recommendation(self) -> State:
        if not self.result:
            # Robustness: nothing came back — apologise gracefully and close.
            self.behaviour.say("I'm sorry, I couldn't find a good match right now.")
            return State.FAREWELL

        top = self.result[0]
        top_event = str(top.get("event", "an event"))
        # Open-palm gesture and the intro line start together and end together.
        self.behaviour.say_with_gesture(
            "Based on what you told me, I recommend the following.", "present")
        self.behaviour.present(self.result)  # presents top events (WP4)
        try:
            reason = self.recommender.explain(top_event, self.evidence)
        except Exception:
            reason = ""
        if reason:
            self.behaviour.say(reason)

        self._wait_for_ack()
        return State.FAREWELL

    def _farewell(self) -> State:
        # Wave and speak the goodbye together, waiting for both to finish.
        self.behaviour.say_with_gesture("Have a great time! Goodbye.", "wave")
        self._reset_interaction()
        return State.IDLE

    # -- acknowledgement / timeout -----------------------------------------

    def _wait_for_ack(self) -> None:
        """Wait until the user acknowledges (leaves) or the timeout elapses."""
        deadline = time.time() + self.ack_timeout
        while self._running and time.time() < deadline:
            self.perception.tick()
            if not self.perception.is_present():
                log.info("User acknowledged / left -> closing.")
                return
            time.sleep(self.idle_poll)
