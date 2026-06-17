"""Local speech-to-text (WP5) using faster-whisper.

Runs entirely on-device (no cloud). Records a microphone utterance with
SpeechRecognition's endpointing (stops on silence), then transcribes it with a
local Whisper model via CTranslate2 — GPU if available, else CPU.

Config via env:
    WHISPER_MODEL    tiny | base | small | medium | large-v3   (default: small)
    WHISPER_DEVICE   auto | cuda | cpu                          (default: auto)
    WHISPER_COMPUTE  float16 | int8_float16 | int8 | ...        (default: by device)
"""

from __future__ import annotations

import io
import logging
import os

log = logging.getLogger("wp5.asr")

_transcriber = None  # process-wide singleton (loading the model is expensive)


def _pick_device(device: str, compute: str):
    if device and device != "auto":
        return device, (compute or ("float16" if device == "cuda" else "int8"))
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", (compute or "float16")
    except Exception:
        pass
    return "cpu", (compute or "int8")


class WhisperTranscriber:
    def __init__(self, model_size=None, device=None, compute=None) -> None:
        from faster_whisper import WhisperModel

        model_size = model_size or os.getenv("WHISPER_MODEL", "small")
        device, compute = _pick_device(
            device or os.getenv("WHISPER_DEVICE", "auto"),
            compute or os.getenv("WHISPER_COMPUTE", ""),
        )
        log.info("Loading Whisper '%s' on %s (%s)...", model_size, device, compute)
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute)
        except Exception as exc:  # GPU libs missing etc. -> fall back to CPU
            log.warning("Whisper on %s failed (%s); falling back to CPU.", device, exc)
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        segments, _info = self.model.transcribe(
            io.BytesIO(wav_bytes), language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def record_utterance(phrase_time_limit: float = 12.0) -> bytes:
    """Capture one spoken utterance from the default mic as WAV bytes."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit)
    return audio.get_wav_data()


def listen_and_transcribe(phrase_time_limit: float = 12.0) -> str:
    """Record one utterance and return its local-Whisper transcript ('' on failure)."""
    try:
        wav = record_utterance(phrase_time_limit)
        return get_transcriber().transcribe_wav(wav)
    except Exception as exc:  # pragma: no cover - hardware/model dependent
        log.warning("ASR failed (%s).", exc)
        return ""
