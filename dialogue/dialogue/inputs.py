"""Input providers — how the dialogue obtains the user's answers.

All implement the same tiny interface so the graph is agnostic to the source:
    ask(question: str) -> str     # present the question, return the user's reply
    notify(message: str) -> None  # say a clarification / status line

  * TypedInput    — console typing (default).
  * SpeechInput   — microphone ASR via SpeechRecognition (Google backend);
                    optionally speaks questions aloud with pyttsx3.
  * ScriptedInput — pre-baked answers for offline tests.

This is also the seam where the robot's ASR/TTS will plug in for WP5: replace the
provider, leave the graph untouched.
"""

from __future__ import annotations

import logging

log = logging.getLogger("wp2.inputs")


class TypedInput:
    def ask(self, question: str) -> str:
        print("\n[robot] " + question)
        try:
            return input("[you ] ").strip()
        except EOFError:
            return ""

    def notify(self, message: str) -> None:
        print("[robot] " + message)


class InterruptInput:
    """Input provider for LangGraph Studio / API runs.

    Instead of blocking on the console, it raises a LangGraph ``interrupt`` so the
    graph pauses and surfaces the question in Studio; the run resumes when you
    submit an answer (Command(resume="...")). This is the same human-in-the-loop
    pattern the robot bridge will use in WP5.
    """

    def ask(self, question: str) -> str:
        from langgraph.types import interrupt
        return interrupt({"question": question})

    def notify(self, message: str) -> None:
        # No interactive channel for status lines in Studio; the next (anchored)
        # question re-asks anyway.
        pass


class ScriptedInput:
    """Returns queued answers in order; used by tests."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._i = 0

    def ask(self, question: str) -> str:
        ans = self._answers[self._i] if self._i < len(self._answers) else ""
        self._i += 1
        return ans

    def notify(self, message: str) -> None:
        pass


class SpeechInput:
    """Microphone speech-to-text. Speaks questions aloud if pyttsx3 is available."""

    def __init__(self, speak_questions: bool = True, phrase_time_limit: float = 8.0) -> None:
        import speech_recognition as sr  # imported here so text mode needs no audio deps

        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._mic = sr.Microphone()
        self._phrase_time_limit = phrase_time_limit
        with self._mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

        self._tts = None
        if speak_questions:
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
            except Exception as exc:  # pragma: no cover
                log.warning("Question TTS unavailable (%s).", exc)

    def _speak(self, text: str) -> None:
        print("[robot] " + text)
        if self._tts is not None:
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception:
                pass

    def ask(self, question: str) -> str:
        self._speak(question)
        print("[robot] (listening...)")
        try:
            with self._mic as source:
                audio = self._recognizer.listen(
                    source, phrase_time_limit=self._phrase_time_limit)
            text = self._recognizer.recognize_google(audio)
            print("[you ] " + text)
            return text
        except self._sr.UnknownValueError:
            print("[robot] (didn't catch that)")
            return ""
        except Exception as exc:  # pragma: no cover - network/mic dependent
            log.warning("ASR failed (%s).", exc)
            return ""

    def notify(self, message: str) -> None:
        self._speak(message)
