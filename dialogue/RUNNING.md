# Running the dialogue manager & viewing traces

Quick guide to running WP2 and inspecting runs in LangSmith / LangGraph Studio.

## 1. Setup (once)

```bash
cd Project/dialogue
uv sync                 # core graph + LLM + tests
uv sync --extra speech  # optional: microphone ASR (SpeechRecognition + pyaudio)
```

Keys live in `Project/src/.env` (gitignored):

```
GROK_KEY="gsk_..."                  # Groq API key (the LLM)
LANGSMITH_API_KEY="lsv2_..."        # LangSmith key (tracing) — must be this exact name
LANGSMITH_TRACING="true"            # turns tracing on
LANGSMITH_PROJECT="dialogue-wp2"    # project name shown in LangSmith
```

> Common gotcha: the tracing key **must** be named `LANGSMITH_API_KEY`. A different
> name (e.g. `LANGGRAPH_KEY`) is ignored and nothing shows up in LangSmith.

## 2. Run it in the terminal

```bash
# Chat with Groq, typing your answers:
uv run python -m dialogue.cli --llm groq --input text

# Microphone speech (needs `uv sync --extra speech`):
uv run python -m dialogue.cli --llm groq --input speech

# Offline, deterministic (no network / no mic) — for development:
uv run python -m dialogue.cli --llm scripted --input scripted --verbose
```

Answer naturally — one sentence can fill several fields ("somewhere cheap outside
with a few friends"). Add `--no-frame` to use fixed question wording instead of
LLM-generated questions.

## 3. View runs in LangSmith (smith.langchain.com)

Every run (terminal or Studio) is logged automatically when tracing is on.

1. Open **https://smith.langchain.com** (the web UI — *not* `api.smith.langchain.com`,
   which is the machine endpoint and returns 404 in a browser).
2. Sign in with the account your `lsv2_...` key belongs to.
3. Left sidebar → **Tracing Projects** → open **`dialogue-wp2`**.
4. Click a trace to expand the run tree:

   ```
   ▾ Router → QuestionFramer → AnswerParser → ConflictResolver → Evaluator → ...
        └─ ChatOpenAI (Groq)   ← the exact prompt + response, tokens, latency
   ```

Each node shows its input/output state (`evidence`, `target`, `feedback`,
`conflict`), so you can see exactly why the agent asked, reframed, or resolved a
conflict.

## 4. Visualize / step through in LangGraph Studio

```bash
cd Project/dialogue
uv run langgraph dev          # opens the Studio web UI
```

Studio shows the **graph topology** and lets you run it in the browser; because
the graph uses `interrupt()`, it pauses at each question and you type the answer
in Studio's resume box. Good for *watching the graph live*; the terminal is
smoother for an actual back-and-forth chat.

> Terminal vs Studio: the terminal CLI runs the graph in its own process — those
> runs appear in **LangSmith** (as traces), not in the live Studio canvas. To
> drive a run *inside* the Studio canvas, start it from Studio.
