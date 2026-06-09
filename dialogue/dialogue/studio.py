"""LangGraph Studio entry point.

Exposes a compiled ``graph`` for `langgraph dev` (the LangGraph Studio web UI).
Uses ``InterruptInput`` so each question pauses the run and surfaces in Studio;
you type the answer there and the graph resumes.

If a Groq key is present it uses the real LLM parser + question framer; otherwise
it falls back to the offline scripted parser so the graph still loads and renders.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the shared key so Studio can call Groq.
load_dotenv(Path(__file__).resolve().parents[2] / "src" / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from dialogue.graph import build_graph
from dialogue.inputs import InterruptInput
from dialogue.llm import ScriptedParser

if os.getenv("GROQ_API_KEY") or os.getenv("GROK_KEY"):
    from dialogue.llm import GroqEvidenceParser, GroqQuestionFramer

    _parser = GroqEvidenceParser()
    _framer = GroqQuestionFramer()
else:  # no key -> offline-renderable fallback
    _parser = ScriptedParser()
    _framer = None

# `langgraph dev` provides the checkpointer; build_graph compiles without one.
graph = build_graph(_parser, InterruptInput(), _framer)
