"""Standalone test harness for the WP2 dialogue manager.

Examples:
    # Offline, deterministic (no network, no mic):
    uv run python -m dialogue.cli --llm scripted --input scripted

    # Real Groq parsing, typed answers:
    uv run python -m dialogue.cli --llm groq --input text

    # Real Groq parsing, microphone speech:
    uv run python -m dialogue.cli --llm groq --input speech
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from dialogue.inputs import ScriptedInput, TypedInput
from dialogue.llm import GroqEvidenceParser, GroqQuestionFramer, ScriptedParser
from dialogue.manager import DialogueManager

# Load the API key. The shared key lives in Project/src/.env; a local .env wins.
load_dotenv(Path(__file__).resolve().parents[2] / "src" / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# A short scripted conversation for offline demos. The first answer deliberately
# fills several slots at once (Budget + Setting + GroupSize).
DEMO_ANSWERS = [
    "somewhere cheap outside with a few friends",
    "something relaxed",
    "in the evening",
    "i'm really into food",
]


def build_parser(name: str):
    if name == "groq":
        return GroqEvidenceParser()
    return ScriptedParser()


def build_framer(llm: str, enabled: bool):
    # LLM-generated questions only make sense with the LLM backend.
    if llm == "groq" and enabled:
        return GroqQuestionFramer()
    return None


def build_input(name: str):
    if name == "text":
        return TypedInput()
    if name == "speech":
        from dialogue.inputs import SpeechInput
        return SpeechInput()
    return ScriptedInput(DEMO_ANSWERS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WP2 dialogue manager test harness.")
    ap.add_argument("--llm", choices=["groq", "scripted"], default="scripted")
    ap.add_argument("--input", choices=["text", "speech", "scripted"], default="scripted")
    ap.add_argument("--no-frame", action="store_true",
                    help="Disable LLM-generated questions (use fixed wording).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    manager = DialogueManager(
        build_parser(args.llm),
        build_input(args.input),
        framer=build_framer(args.llm, not args.no_frame),
    )

    print("=== WP2 Dialogue Manager (llm=%s, input=%s) ===" % (args.llm, args.input))
    state = manager.run_verbose()
    print("\n--- Collected evidence ---")
    for slot, value in state["evidence"].items():
        print("  %-14s %s" % (slot, value))
    missing = [s for s in ("Budget", "GroupSize", "ActivityLevel",
                           "Setting", "TimeOfDay", "Interest")
               if s not in state["evidence"]]
    if missing:
        print("  (missing: %s)" % ", ".join(missing))
    if args.verbose:
        print("\n--- Transcript ---")
        for turn in state["turn_log"]:
            print("  Q: %s" % turn["q"])
            print("  A: %s" % turn["a"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
