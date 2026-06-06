"""Persistent text-to-speech worker (WP4).

Keeps a single pyttsx3 engine alive on a dedicated thread and speaks queued
utterances using the startLoop(False)/iterate() pattern. This avoids two
problems with the naive approach:

  * re-initialising the engine per utterance (adds ~100-300 ms latency before
    every line, which desyncs speech from gestures), and
  * reusing one engine with repeated runAndWait() calls, which raises
    "run loop already started" so only the first line is ever spoken.

``speak()`` blocks by default until the utterance finishes (via the
'finished-utterance' callback), so callers can coordinate gestures with speech.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

log = logging.getLogger("wp4.tts")

try:
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    pyttsx3 = None  # type: ignore


class TtsEngine:
    def __init__(self) -> None:
        self.available = pyttsx3 is not None
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._done = None  # completion event for the utterance in progress
        self._thread = None
        if self.available:
            self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
            self._thread.start()

    def _on_finished(self, name, completed) -> None:
        d = self._done
        if d is not None:
            d.set()

    def _run(self) -> None:
        try:
            engine = pyttsx3.init()
        except Exception as exc:  # pragma: no cover
            log.warning("TTS init failed (%s); speech disabled.", exc)
            self.available = False
            self._drain()
            return

        engine.connect("finished-utterance", self._on_finished)
        engine.startLoop(False)
        try:
            while not self._stop.is_set():
                try:
                    text, done = self._queue.get(timeout=0.05)
                except queue.Empty:
                    engine.iterate()  # keep the loop alive
                    continue
                self._done = done
                engine.say(text)
                while not done.is_set() and not self._stop.is_set():
                    engine.iterate()
                    time.sleep(0.005)
                self._done = None
        finally:
            try:
                engine.endLoop()
            except Exception:
                pass

    def _drain(self) -> None:
        # If TTS is unavailable, release any blocked callers so they don't hang.
        while not self._stop.is_set():
            try:
                _text, done = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            done.set()

    def speak(self, text: str, block: bool = True, timeout: float = 20.0) -> None:
        if not self.available:
            return
        done = threading.Event()
        self._queue.put((text, done))
        if block:
            done.wait(timeout=timeout)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
