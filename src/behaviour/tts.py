"""Text-to-speech worker (WP4) with a neural voice and a robust fallback.

Speech is queued and spoken on a dedicated thread so it can run alongside
gestures. Backends, in order of preference (override with TTS_BACKEND):

  * edge  — Microsoft Edge neural voices (natural prosody; needs internet +
            `edge-tts` and `playsound`). Voice via TTS_VOICE
            (default en-US-AriaNeural).
  * pyttsx3 — offline SAPI5/espeak fallback (more monotone, always available).

If the neural backend errors at runtime (e.g. offline), the worker falls back to
pyttsx3 for that line, so speech never silently breaks.
"""

from __future__ import annotations

import logging
import os
import queue
import threading

log = logging.getLogger("wp4.tts")


# ---------------------------------------------------------------------------
# Audio playback (for file-based backends)
# ---------------------------------------------------------------------------

def _play_audio(path: str) -> None:
    try:
        from playsound import playsound
        playsound(path, True)
        return
    except Exception as exc:
        if path.lower().endswith(".wav"):
            try:
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
                return
            except Exception:
                pass
        raise exc


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _EdgeBackend:
    name = "edge-tts"

    def __init__(self, voice: str) -> None:
        self.voice = voice

    @classmethod
    def try_create(cls):
        try:
            import edge_tts  # noqa: F401
        except Exception:
            return None
        try:
            import playsound  # noqa: F401  (mp3 playback)
        except Exception:
            log.warning("edge-tts present but 'playsound' missing; cannot play audio.")
            return None
        return cls(os.getenv("TTS_VOICE", "en-US-AriaNeural"))

    def speak(self, text: str) -> None:
        import asyncio
        import tempfile

        import edge_tts

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            async def _synth():
                await edge_tts.Communicate(text, self.voice).save(tmp.name)

            asyncio.run(_synth())
            _play_audio(tmp.name)
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    def close(self) -> None:
        pass


class _Pyttsx3Backend:
    name = "pyttsx3"

    @classmethod
    def try_create(cls):
        try:
            import pyttsx3  # noqa: F401
        except Exception:
            return None
        return cls()

    def speak(self, text: str) -> None:
        # Fresh engine per utterance: reliable across repeated calls (avoids the
        # "run loop already started" issue when reusing one engine).
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def close(self) -> None:
        pass


def _select_backend():
    pref = os.getenv("TTS_BACKEND", "auto").lower()
    if pref == "none":
        return None, None
    primary = None
    if pref in ("auto", "edge"):
        primary = _EdgeBackend.try_create()
    if primary is None and pref in ("auto", "pyttsx3"):
        primary = _Pyttsx3Backend.try_create()
    # A pyttsx3 fallback used if the neural backend errors at runtime.
    fallback = None
    if primary is not None and primary.name != "pyttsx3":
        fallback = _Pyttsx3Backend.try_create()
    return primary, fallback


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TtsEngine:
    def __init__(self) -> None:
        self._backend, self._fallback = _select_backend()
        self.available = self._backend is not None
        if self.available:
            log.info("TTS backend: %s", self._backend.name)
        else:
            log.warning("No TTS backend available; speech disabled.")
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if self.available:
            self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text, done = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._backend.speak(text)
            except Exception as exc:
                log.warning("TTS (%s) failed (%s).", self._backend.name, exc)
                if self._fallback is not None:
                    try:
                        self._fallback.speak(text)
                    except Exception as exc2:  # pragma: no cover
                        log.warning("TTS fallback failed (%s).", exc2)
            finally:
                done.set()
        for backend in (self._backend, self._fallback):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass

    def speak(self, text: str, block: bool = True, timeout: float = 30.0) -> None:
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
