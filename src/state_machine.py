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
    AGENT_NAME,
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
        get_text: Optional[Callable[[], str]] = None,
        enroll_samples: int = 15,
    ) -> None:
        self.perception = perception
        self.dialogue = dialogue
        self.recommender = recommender
        self.behaviour = behaviour

        self.idle_poll = idle_poll
        self.presence_check_frames = presence_check_frames
        self.ack_timeout = ack_timeout
        self.name_provider = name_provider
        # Optional requirement #2: asks a not-yet-authorized visitor's name and
        # (with consent) enrolls their face. get_text supplies typed replies.
        self.get_text = get_text or self._default_get_text
        self.enroll_samples = enroll_samples

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

    @staticmethod
    def _default_get_text() -> str:
        try:
            return input("  [you] ").strip()
        except EOFError:
            return ""

    def _capture_enrollment_samples(self, recognizer, name: str) -> int:
        """Grab a few live frames of the person currently in view and save
        them as enrollment samples. Reuses the already-open camera feed
        (``self.perception``) rather than opening a second one."""
        saved = 0
        attempts = 0
        max_attempts = self.enroll_samples * 4
        while saved < self.enroll_samples and attempts < max_attempts:
            self.perception.tick()
            attempts += 1
            frame = getattr(self.perception, "last_frame", None)
            faces = getattr(self.perception, "last_faces", None)
            if frame is not None and faces is not None and len(faces):
                box = max(faces, key=lambda b: b[2] * b[3])  # largest face
                try:
                    recognizer.enroll_sample(frame, box, name)
                    saved += 1
                except Exception:
                    log.debug("Enrollment sample capture failed.", exc_info=True)
            time.sleep(0.15)
        return saved

    def _identify_or_enroll(self) -> Optional[str]:
        """Optional requirements #1/#2: verify the face; if unrecognized, ask
        for a name and (with consent) enroll it for next time.

        Returns the person's name if known, freshly enrolled, or given
        without consent to save; None if nothing was learned (e.g. no camera
        source is wired, or the visitor declines to give a name).
        """
        identify = getattr(self.perception, "identify", None)
        if identify is not None:
            identity = identify()
            if identity:
                return identity

        recognizer = getattr(self.perception, "recognizer", None)
        if recognizer is None:
            return None  # no camera / no recognizer wired for this source

        self.behaviour.say("I don't think we've met yet. What's your name?")
        name = (self.get_text() or "").strip()
        if not name:
            return None

        self.behaviour.say(
            "Nice to meet you, %s. Would it be okay if I remember your "
            "face for next time?" % name
        )
        consent = (self.get_text() or "").strip().lower()
        if consent.startswith("y"):
            saved = self._capture_enrollment_samples(recognizer, name)
            if saved:
                recognizer.retrain()
                self.behaviour.say("Great, I'll remember you next time, %s!" % name)
            else:
                self.behaviour.say(
                    "Hmm, I couldn't get a clear look — no worries, let's continue.")
        else:
            self.behaviour.say("No problem, %s." % name)
        return name

    def _reset_interaction(self) -> None:
        clear_display = getattr(self.behaviour, "clear_display", None)
        if clear_display is not None:
            try:
                clear_display()
            except Exception:
                log.debug("Behaviour display reset failed.", exc_info=True)
        self.evidence = {}
        self.result = []
        self.perception.reset()

    # -- state handlers (each returns the next State) -----------------------

    def _idle(self) -> State:
        # Passive monitoring until a face is stable for several frames.
        while self._running:
            self.perception.tick()
            if self.perception.face_stable():
                # Optional requirement #1: verify the face against authorized
                # users (logged here; _greeting() also uses it to personalize
                # or to ask+enroll an unrecognized visitor).
                identify = getattr(self.perception, "identify", None)
                identity = identify() if identify is not None else None
                if identity:
                    log.info("Stable face detected -> Greeting (authorized user: %s)",
                             identity)
                else:
                    log.info("Stable face detected -> Greeting (unrecognized visitor)")
                return State.GREETING
            time.sleep(self.idle_poll)
        return State.IDLE

    def _greeting(self) -> State:
        name = self._identify_or_enroll()
        if name is None and self.name_provider is not None:
            try:
                name = self.name_provider()
            except Exception:
                name = None
        hello = ("Greetings, %s. %s online, at your service." % (name, AGENT_NAME) if name
                 else "Greetings. %s online, at your service." % AGENT_NAME)
        # Two coordinated beats: each line and its gesture start together and
        # the FSM waits for both to finish before the next.
        self.behaviour.say_with_gesture(hello, "wave")
        self.behaviour.say_with_gesture("Systems nominal. It's good to see you.", "nod")
        self.behaviour.say_with_gesture(
            "Scanning the local grid for a social event to suit you.", "open_arms")

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
        self.behaviour.say_with_gesture(
            "Enjoy your evening. %s signing off, goodbye." % AGENT_NAME, "wave")
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
