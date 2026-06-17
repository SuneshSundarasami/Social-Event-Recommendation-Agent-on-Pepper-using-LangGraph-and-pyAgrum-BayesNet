# Social Event Recommendation Agent

A socially interactive event recommendation agent for the **Pepper** robot in the
**qiBullet** simulation. It detects a nearby user, runs a short conversation to
elicit activity preferences, reasons over them with a Bayesian network, and
recommends suitable social events.

See [PLAN.md](PLAN.md) for the full project plan, work packages, shared
contracts, and design notes.

## Status

| WP | Component | State |
|----|-----------|-------|
| WP1 | Perception + interaction FSM | ✅ implemented (`src/`) |
| WP2 | Dialogue manager (LangGraph + Groq) | ✅ implemented (`dialogue/`) |
| WP3 | Bayesian recommender (pyAgrum) | ✅ implemented (`src/recommender/`) |
| WP4 | Behaviour layer (speech + gestures) | ✅ initial (`src/behaviour/`) |
| WP5 | Integration (3.8↔3.11 dialogue bridge) | ✅ working (`src/dialogue_bridge.py`); demo tuning ongoing |

> WP2 runs as its own Python 3.11 uv project under [dialogue/](dialogue/) (LangGraph
> needs ≥3.9). See [dialogue/README.md](dialogue/README.md) and
> [dialogue/RUNNING.md](dialogue/RUNNING.md) for running it and viewing LangSmith traces.

## Setup

This is a standalone [uv](https://docs.astral.sh/uv/) project pinned to Python
3.8 (NAOqi / qiBullet compatibility).

```bash
cd Project
uv sync
```

## Running WP1

```bash
# Headless demo — no camera / no qiBullet, one full interaction cycle:
uv run python src/main.py --source scripted

# Local webcam face detection + interactive console dialogue:
uv run python src/main.py --source webcam --interactive

# Pepper's simulated top camera in qiBullet:
uv run python src/main.py --source pepper

# Full pipeline with the real WP2 LangGraph dialogue (Pepper speaks each
# question, you type the answer) -> Bayesian recommendation:
uv run python src/main.py --source scripted --dialogue real

# Fully spoken: Pepper asks out loud (neural TTS), you answer out loud
# (local Whisper ASR), Pepper in the qiBullet GUI:
uv run python src/main.py --source hybrid --dialogue real --answer speech
```

> `--dialogue real` spawns the WP2 dialogue service (Python 3.11 venv under
> [dialogue/](dialogue/)) as a subprocess and talks to it over JSON stdio. Run
> `uv sync` (robot) and `uv sync --extra speech` (in `dialogue/`, for `--answer
> speech`) first.

### Voice config (env)

| Variable | Default | Notes |
|---|---|---|
| `TTS_BACKEND` | `auto` | `edge` (neural), `pyttsx3` (offline), or `none` |
| `TTS_VOICE` | `en-US-AriaNeural` | any Edge neural voice |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` (6 GB VRAM fits up to ~medium) |
| `WHISPER_DEVICE` | `auto` | `cuda` / `cpu` (auto-detects GPU) |

Neural TTS (`edge-tts`) is online; local Whisper ASR runs on-device (GPU if
available). To go fully offline, set `TTS_BACKEND=pyttsx3`.

## Layout

```
src/
  contracts.py      # shared Evidence/Result types + cross-package interfaces
  state_machine.py  # six-state interaction FSM (WP1)
  perception/       # face detection + camera sources (WP1)
  stubs.py          # console stand-ins for WP2/WP3/WP4
  main.py           # orchestrator / entry point
```
