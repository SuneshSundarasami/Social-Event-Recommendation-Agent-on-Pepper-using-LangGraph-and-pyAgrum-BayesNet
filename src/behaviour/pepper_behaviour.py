"""Pepper behaviour layer (WP4 — initial).

Drives real joint motion on the qiBullet Pepper for the interaction's gestures
(wave / nod / think / present) and speaks via local TTS (pyttsx3) with a console
fallback. Satisfies the ``Behaviour`` contract so the FSM can use it in place of
the ConsoleBehaviour stub.

Gesture joint poses are adapted from the HW_02 behaviour plugins.
"""

from __future__ import annotations

import logging
import threading
import time

from behaviour.event_display import EventImageDisplay
from behaviour.tts import TtsEngine

log = logging.getLogger("wp4.behaviour")


class PepperBehaviour:
    def __init__(self, pepper, speak_aloud: bool = True) -> None:
        self.pepper = pepper
        self._gesture_thread = None  # most recent gesture, runs concurrently with speech
        self._tts = TtsEngine() if speak_aloud else None
        self._display = EventImageDisplay(pepper)

    # -- speech -------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop speech worker and wait for any in-flight gesture (before sim stop)."""
        self.wait_for_gesture()
        self._display.close()
        if self._tts is not None:
            self._tts.shutdown()

    def clear_display(self) -> None:
        self._display.clear()

    def say(self, text: str) -> None:
        print("  [Pepper] (says)    %s" % text)
        if self._tts is not None:
            self._tts.speak(text)  # blocks until the line finishes

    # -- gestures -----------------------------------------------------------

    def gesture(self, name: str) -> None:
        """Start a gesture and return immediately so it plays *alongside* speech.

        Gestures are serialized among themselves (a new one waits for the
        previous to finish) but run concurrently with say(), keeping the robot's
        motion in sync with the voice instead of one-after-the-other.
        """
        handler = {
            "wave": self._wave,
            "nod": self._nod,
            "think": self._think,
            "present": self._present_gesture,
            "open_arms": self._open_arms,
            "talk": self._talk,
        }.get(name)
        if handler is None:
            print("  [Pepper] (gesture) %s (no joint mapping — skipped)" % name)
            return
        print("  [Pepper] (gesture) %s" % name)
        self.wait_for_gesture()  # don't overlap two joint sequences

        def _run():
            try:
                handler()
            except Exception as exc:  # pragma: no cover
                log.warning("Gesture %r failed (%s).", name, exc)

        self._gesture_thread = threading.Thread(target=_run, daemon=True)
        self._gesture_thread.start()

    def wait_for_gesture(self) -> None:
        """Block until any in-flight gesture finishes."""
        t = self._gesture_thread
        if t is not None and t.is_alive():
            t.join()

    def say_with_gesture(self, text: str, name: str) -> None:
        """Start speech and gesture together; return only once both have ended."""
        self.gesture(name)          # launches the motion on a thread
        self.say(text)              # speaks (blocks on TTS) concurrently
        self.wait_for_gesture()     # ensure the motion has also finished

    def _wave(self) -> None:
        p = self.pepper
        # Raise both arms to neutral shoulder height.
        p.setAngles(["LShoulderPitch", "LShoulderRoll", "LElbowRoll"],
                    [1.4, 0.1, -0.5], 0.5)
        p.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                    [1.4, -0.1, 0.5], 0.5)
        time.sleep(0.8)
        # Extend the right arm up into a waving position.
        p.setAngles("RShoulderPitch", -0.5, 0.5)
        p.setAngles("RShoulderRoll", -0.5, 0.5)
        p.setAngles("RElbowRoll", 1.0, 0.5)
        time.sleep(1.0)
        # Oscillate the elbow to wave.
        for _ in range(3):
            p.setAngles("RElbowRoll", 0.5, 0.8)
            time.sleep(0.4)
            p.setAngles("RElbowRoll", 1.2, 0.8)
            time.sleep(0.4)
        self._relax_arms()

    def _nod(self) -> None:
        p = self.pepper
        for _ in range(2):
            p.setAngles("HeadPitch", 0.35, 0.3)
            time.sleep(0.4)
            p.setAngles("HeadPitch", -0.1, 0.3)
            time.sleep(0.4)
        p.setAngles("HeadPitch", 0.0, 0.3)

    def _think(self) -> None:
        p = self.pepper
        # Bring the right hand up toward the chin and tilt the head, pondering.
        p.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"],
                    [0.6, -0.2, 1.5, 1.0], 0.4)
        p.setAngles(["HeadPitch", "HeadYaw"], [0.2, 0.2], 0.3)
        time.sleep(1.5)
        p.setAngles(["HeadPitch", "HeadYaw"], [0.0, 0.0], 0.3)
        self._relax_arms()

    def _present_gesture(self) -> None:
        p = self.pepper
        # Open-palm presentation: right arm extended forward, palm up.
        p.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"],
                    [0.5, -0.2, 0.4, 0.5], 0.4)
        try:
            p.openHand("RHand")
        except Exception:
            pass
        time.sleep(1.5)
        self._relax_arms()

    def _open_arms(self) -> None:
        # Conversational two-handed gesture: forearms raised so the open hands
        # sit near the shoulders with the elbows kept down by the sides, then a
        # few small up/down swings — like a person gesturing while they talk.
        p = self.pepper
        p.setAngles(["RShoulderPitch", "LShoulderPitch"], [1.0, 1.0], 0.3)
        p.setAngles(["RShoulderRoll", "LShoulderRoll"], [-0.15, 0.15], 0.3)
        p.setAngles(["RElbowRoll", "LElbowRoll"], [1.5, -1.5], 0.3)
        p.setAngles(["RElbowYaw", "LElbowYaw"], [0.3, -0.3], 0.3)
        try:
            p.openHand("RHand")
            p.openHand("LHand")
        except Exception:
            pass
        time.sleep(0.6)
        # Slight, natural swing of both hands a few times.
        for _ in range(3):
            p.setAngles(["RShoulderPitch", "LShoulderPitch"], [0.9, 0.9], 0.25)
            time.sleep(0.3)
            p.setAngles(["RShoulderPitch", "LShoulderPitch"], [1.05, 1.05], 0.25)
            time.sleep(0.3)
        try:
            p.closeHand("RHand")
            p.closeHand("LHand")
        except Exception:
            pass
        self._relax_arms()

    def _talk(self) -> None:
        # A single, brief conversational beat -- one small raise-and-settle
        # instead of open_arms's repeated swing cycles, so it stays roughly in
        # sync however long the (LLM-generated, variable-length) question is.
        p = self.pepper
        p.setAngles(["RShoulderPitch", "LShoulderPitch"], [1.0, 1.05], 0.35)
        p.setAngles(["RShoulderRoll", "LShoulderRoll"], [-0.15, 0.15], 0.35)
        p.setAngles(["RElbowRoll", "LElbowRoll"], [1.3, -1.2], 0.35)
        try:
            p.openHand("RHand")
        except Exception:
            pass
        time.sleep(0.45)
        try:
            p.closeHand("RHand")
        except Exception:
            pass
        self._relax_arms()

    def _relax_arms(self) -> None:
        p = self.pepper
        p.setAngles(["RShoulderPitch", "LShoulderPitch"], [1.4, 1.4], 0.4)
        p.setAngles(["RShoulderRoll", "LShoulderRoll"], [-0.1, 0.1], 0.4)
        p.setAngles(["RElbowRoll", "LElbowRoll"], [0.5, -0.5], 0.4)
        time.sleep(0.8)

    # -- presentation -------------------------------------------------------

    def present(self, result) -> None:
        print("  [Pepper] (presents top events):")
        if result:
            top_event = str(result[0].get("event", ""))
            image_path = self._display.show_event(top_event)
            if image_path:
                print("  [Pepper] (tablet)  %s" % image_path)
        for i, item in enumerate(result[:3], start=1):
            event = item.get("event", "?")
            prob = item.get("prob", 0.0)
            try:
                prob_str = "%.0f%%" % (float(prob) * 100)
            except Exception:
                prob_str = str(prob)
            print("      %d. %-12s %s" % (i, event, prob_str))
