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


def _register_cuda_dlls() -> None:
    """Add the NVIDIA pip wheels' DLL dirs so CTranslate2 finds cuBLAS/cuDNN.

    On Windows the nvidia-*-cu12 wheels drop their DLLs under
    site-packages/nvidia/<lib>/bin, which isn't on the loader path by default.
    """
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        base = list(spec.submodule_search_locations)[0]
        dirs = []
        for lib in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = os.path.join(base, lib, "bin")
            if os.path.isdir(d):
                dirs.append(d)
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        # CTranslate2 loads cuBLAS/cuDNN via plain LoadLibrary, which searches
        # PATH (not the add_dll_directory set) — so prepend them to PATH too.
        if dirs:
            os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
            log.debug("CUDA DLL dirs on PATH: %s", dirs)
    except Exception as exc:  # pragma: no cover
        log.debug("CUDA DLL registration skipped: %s", exc)


_register_cuda_dlls()


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
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "small")
        device, compute = _pick_device(
            device or os.getenv("WHISPER_DEVICE", "auto"),
            compute or os.getenv("WHISPER_COMPUTE", ""),
        )
        self._load(device, compute)

    def _load(self, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        log.info("Loading Whisper '%s' on %s (%s)...", self.model_size, device, compute)
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute)
        self.device = device

    def _transcribe(self, wav_bytes: bytes) -> str:
        segments, _info = self.model.transcribe(
            io.BytesIO(wav_bytes), language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        try:
            return self._transcribe(wav_bytes)
        except Exception as exc:
            # GPU compute can fail at runtime if CUDA libs (cuBLAS/cuDNN) are
            # missing — fall back to CPU once and stay there.
            if self.device != "cpu":
                log.warning("GPU transcription failed (%s); switching to CPU.", exc)
                self._load("cpu", "int8")
                return self._transcribe(wav_bytes)
            raise


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def record_utterance(phrase_time_limit: float = 12.0, timeout: float = 12.0) -> bytes:
    """Capture one spoken utterance from the mic as WAV bytes.

    Env: WHISPER_MIC_INDEX (input device index), WHISPER_ENERGY (fixed energy
    threshold; otherwise calibrated from ambient noise).
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    idx = os.getenv("WHISPER_MIC_INDEX")
    mic = sr.Microphone(device_index=int(idx)) if idx else sr.Microphone()
    with mic as source:
        energy = os.getenv("WHISPER_ENERGY")
        if energy:
            recognizer.energy_threshold = float(energy)
            recognizer.dynamic_energy_threshold = False
        else:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
        log.info("Listening (energy_threshold=%.0f)...", recognizer.energy_threshold)
        audio = recognizer.listen(source, timeout=timeout,
                                  phrase_time_limit=phrase_time_limit)
    wav = audio.get_wav_data()
    log.info("Captured %d bytes of audio.", len(wav))
    return wav


def listen_and_transcribe(phrase_time_limit: float = 12.0) -> str:
    """Record one utterance and return its local-Whisper transcript ('' on failure)."""
    try:
        wav = record_utterance(phrase_time_limit)
    except Exception as exc:  # WaitTimeoutError, no mic, etc.
        log.warning("Mic capture failed (%s).", exc)
        return ""
    try:
        text = get_transcriber().transcribe_wav(wav)
        log.info("Transcript: %r", text)
        return text
    except Exception as exc:  # pragma: no cover
        log.warning("Whisper transcription failed (%s).", exc)
        return ""


if __name__ == "__main__":
    # Standalone mic + Whisper check:  uv run python -m dialogue.asr
    import speech_recognition as sr

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("Input devices:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print("  [%d] %s" % (i, name))
    print("\nModel:", os.getenv("WHISPER_MODEL", "small"),
          "| device:", os.getenv("WHISPER_DEVICE", "auto"))
    print("Speak after 'Listening...' appears.\n")
    print("HEARD:", repr(listen_and_transcribe()))
