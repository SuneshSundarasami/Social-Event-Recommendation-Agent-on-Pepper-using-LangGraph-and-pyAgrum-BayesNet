"""WP5 bridge — run the dialogue graph as a subprocess driven over JSON stdio.

The Python 3.8 robot process spawns this (Python 3.11) module and exchanges
newline-delimited JSON on stdin/stdout, so the LangGraph dialogue can run behind
the robot's `collect_evidence()` call while the robot owns the voice / input.

Protocol (service <-> robot):
    service -> robot : {"event":"ready"}
                       {"event":"ask","question":"..."}    # robot speaks + answers
                       {"event":"notify","message":"..."}
                       {"event":"evidence","evidence":{...}}
                       {"event":"error","message":"..."}
    robot -> service : {"cmd":"collect"} | {"cmd":"quit"}
                       {"answer":"..."}

Only protocol JSON is written to stdout; all logs go to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Shared key lives in Project/src/.env; a local dialogue/.env wins.
load_dotenv(Path(__file__).resolve().parents[2] / "src" / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

from dialogue.llm import ScriptedParser
from dialogue.manager import DialogueManager


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read():
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except Exception:
        return {}


class RemoteInput:
    """Input provider that asks the robot (over stdio) instead of the console.

    The robot replies either with typed text ({"answer": ...}) or, in speech
    mode, with {"spoken": true} once Pepper has finished voicing the question —
    in which case we record the mic and transcribe it locally with Whisper.
    """

    def ask(self, question: str) -> str:
        _emit({"event": "ask", "question": question})
        while True:
            msg = _read()
            if msg is None:
                raise EOFError("robot closed the connection")
            if "answer" in msg:
                return msg["answer"] or ""
            if msg.get("spoken"):
                return self._listen()

    def _listen(self) -> str:
        from dialogue.asr import listen_and_transcribe
        _emit({"event": "notify", "message": "(listening...)"})
        text = listen_and_transcribe()
        _emit({"event": "heard", "text": text})
        return text

    def notify(self, message: str) -> None:
        _emit({"event": "notify", "message": message})


def _build_llm():
    if os.getenv("GROQ_API_KEY") or os.getenv("GROK_KEY"):
        from dialogue.llm import GroqEvidenceParser, GroqQuestionFramer
        return GroqEvidenceParser(), GroqQuestionFramer()
    return ScriptedParser(), None


def main() -> None:
    parser, framer = _build_llm()
    _emit({"event": "ready"})
    while True:
        msg = _read()
        if msg is None:
            break
        cmd = msg.get("cmd")
        if cmd == "quit":
            break
        if cmd == "collect":
            manager = DialogueManager(parser, RemoteInput(), framer=framer)
            try:
                evidence = manager.collect_evidence()
                _emit({"event": "evidence", "evidence": evidence})
            except Exception as exc:  # pragma: no cover
                _emit({"event": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
