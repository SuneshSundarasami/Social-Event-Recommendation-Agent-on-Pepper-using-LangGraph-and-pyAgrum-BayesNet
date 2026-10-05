# WP2 — Dialogue Manager (LangGraph + Groq)

Standalone dialogue service for the Social Event Recommendation Agent. It asks
the user for their event preferences, parses free-form answers into discrete
evidence with an LLM, and returns the validated `Evidence` dict the Bayesian
recommender (WP3) consumes.

Runs in **its own Python 3.11 uv project** because LangGraph needs ≥3.9, while
the robot side (perception/behaviour) is pinned to 3.8. The robot calls it
through one method: `DialogueManager.collect_evidence() -> Evidence`. The 3.8↔3.11
process bridge is deferred to WP5.

## Graph

```
select_slot -> ask_question -> parse_llm -> validate -> (commit | clarify) -> ...
```

An evidence-driven loop driven by [slots.py](dialogue/slots.py): look at what's
still unknown → ask → LLM parse (may fill several slots at once) → whitelist &
commit → clarify ambiguous answers, falling back to a default after N tries. See
[PLAN.md §5](../PLAN.md) for the design.

**Conversational questions.** With Groq, the next question is *generated* each
turn from what's known vs. still missing (`GroqQuestionFramer`) — fresh wording,
no form-like option lists, and one rich answer ("somewhere cheap outside with a
few friends") fills several slots so those questions are skipped. A randomly
chosen style hint (calm curiosity, a brief everyday comparison, an occasional
understated touch of dry wit, …) is nudged into the prompt each call so
repeated turns don't converge on one template; the tone overall stays
composed and professional rather than jokey. Use `--no-frame` to fall back to
fixed question wording.

**Slot order.** `Router` (in `graph.py`) picks the next unknown slot at
random each run by default (`randomize_order=True` in `build_graph` /
`DialogueManager`), so the conversation doesn't always open with the same
question (e.g. always Budget). Tests that hard-code answers against a fixed
sequence pass `randomize_order=False`.

**Abusive language.** `AnswerParser` checks the raw reply against a small
flagged-word list (`_ABUSE_WORDS` in `graph.py`) before it reaches the LLM
parser; a match short-circuits to a polite "let's keep this friendly" prompt
instead of being parsed as evidence.

## Setup

```bash
cd Project/dialogue
uv sync                 # core graph + LLM + tests
uv sync --extra speech  # add microphone ASR (SpeechRecognition + pyaudio)
```

### Groq API key

The evidence parser uses Groq (free open models, OpenAI-compatible API). Put the
key in `Project/src/.env` (shared) or a local `Project/dialogue/.env`:

```
GROK_KEY=gsk_...                    # or GROQ_API_KEY; from https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile  # optional; any free Groq model works
```

> If the key/model is wrong the parser logs a warning and returns no evidence
> (the graph then clarifies / falls back to defaults).

## Run

```bash
# Offline, deterministic (no network, no mic) — great for development:
uv run python -m dialogue.cli --llm scripted --input scripted --verbose

# Real Groq parsing, typed answers:
uv run python -m dialogue.cli --llm groq --input text

# Real Groq parsing, microphone speech (needs --extra speech):
uv run python -m dialogue.cli --llm groq --input speech
```

## Test

```bash
uv run pytest -q
```

## Layout

```
dialogue/
  schema.py    # evidence labels + whitelist guard (mirrors src/contracts.py)
  slots.py     # the question plan (questions, clarifications, defaults)
  llm.py       # GroqEvidenceParser + ScriptedParser (offline/fallback)
  inputs.py    # TypedInput / SpeechInput / ScriptedInput providers
  graph.py     # the LangGraph StateGraph
  manager.py   # DialogueManager.collect_evidence()
  cli.py       # test harness
tests/         # offline graph tests
```
