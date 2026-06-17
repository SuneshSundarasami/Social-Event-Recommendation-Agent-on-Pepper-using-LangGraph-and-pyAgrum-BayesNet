"""Robot-side client for the WP2 dialogue service (WP5 integration).

Implements the ``DialogueManager`` contract (`collect_evidence() -> Evidence`) on
the Python 3.8 robot side by spawning the Python 3.11 dialogue service (its own
uv venv) and driving it over JSON stdio. For each question the dialogue asks,
``ask_user(question)`` is invoked — the robot speaks it and returns the reply.

Robust by design: any failure (missing venv, crashed service, bad answer) is
caught and yields empty evidence, so the Bayesian recommender still runs on
whatever was gathered (proposal §6).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from typing import Callable, Optional

from contracts import Evidence

log = logging.getLogger("wp5.bridge")


class DialogueBridge:
    def __init__(self, speak: Callable[[str], None], dialogue_dir: str,
                 answer_mode: str = "text", get_text: Optional[Callable[[], str]] = None,
                 notify: Optional[Callable[[str], None]] = None,
                 on_heard: Optional[Callable[[str], None]] = None) -> None:
        # speak: voice a question (Pepper). answer_mode: "text" -> get_text() supplies
        # the typed reply; "speech" -> the service records the mic + Whisper transcribes.
        self.speak = speak
        self.answer_mode = answer_mode
        self.get_text = get_text
        self.notify = notify
        self.on_heard = on_heard
        self.dialogue_dir = dialogue_dir
        self.proc = self._spawn(dialogue_dir)
        self._drain_stderr()
        self._await_ready()

    # -- process management -------------------------------------------------

    @staticmethod
    def _service_cmd(dialogue_dir: str):
        win = os.path.join(dialogue_dir, ".venv", "Scripts", "python.exe")
        posix = os.path.join(dialogue_dir, ".venv", "bin", "python")
        if os.path.exists(win):
            return [win, "-m", "dialogue.bridge"]
        if os.path.exists(posix):
            return [posix, "-m", "dialogue.bridge"]
        uv = shutil.which("uv")
        if uv:
            return [uv, "run", "python", "-m", "dialogue.bridge"]
        raise RuntimeError("dialogue venv not found and 'uv' not on PATH "
                           "(run `uv sync` in the dialogue project)")

    def _spawn(self, dialogue_dir: str):
        cmd = self._service_cmd(dialogue_dir)
        log.info("Starting dialogue service: %s (cwd=%s)", " ".join(cmd), dialogue_dir)
        return subprocess.Popen(
            cmd, cwd=dialogue_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def _drain_stderr(self) -> None:
        # Read the service's stderr in the background so it can't fill the pipe
        # and deadlock; surface it at debug level.
        def run():
            for line in self.proc.stderr:
                log.debug("[dialogue] %s", line.rstrip())
        threading.Thread(target=run, daemon=True).start()

    # -- protocol -----------------------------------------------------------

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("dialogue service exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue  # ignore any non-protocol noise on stdout

    def _await_ready(self) -> None:
        msg = self._recv()
        if msg.get("event") != "ready":
            raise RuntimeError("unexpected handshake from dialogue service: %r" % msg)

    # -- DialogueManager contract ------------------------------------------

    def collect_evidence(self) -> Evidence:
        try:
            self._send({"cmd": "collect"})
            while True:
                msg = self._recv()
                event = msg.get("event")
                if event == "ask":
                    self.speak(msg.get("question", ""))   # Pepper voices the question
                    if self.answer_mode == "speech":
                        # Service records the mic + transcribes locally with Whisper.
                        self._send({"spoken": True})
                    else:
                        answer = self.get_text() if self.get_text else ""
                        self._send({"answer": answer or ""})
                elif event == "heard":
                    if self.on_heard:
                        self.on_heard(msg.get("text", ""))
                elif event == "notify":
                    if self.notify:
                        self.notify(msg.get("message", ""))
                elif event == "evidence":
                    return msg.get("evidence", {}) or {}
                elif event == "error":
                    log.warning("dialogue service error: %s", msg.get("message"))
                    return {}
        except Exception as exc:
            log.warning("dialogue bridge failed (%s); continuing with no evidence.", exc)
            return {}

    def close(self) -> None:
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
