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

### Tablet event images

When Pepper presents a recommendation in qiBullet, the behaviour layer maps the
top-ranked event to an image in [imgs/](imgs/) and displays it on a visual panel
aligned with Pepper's tablet. The panel is cleared at the end of each interaction
cycle, so the next user starts from an empty tablet.

The event image mapping lives in `src/behaviour/event_display.py`. Current assets:

| Event | Image |
|---|---|
| Museum | `imgs/mueseum.jpg` |
| Concert | `imgs/concert.jpg` |
| Sports | `imgs/sports.jpg` |
| Food | `imgs/food.avif` |
| Outdoor | `imgs/outdoor.jpg` |
| Nightlife | `imgs/nightlife.jpg` |
| Workshop | `imgs/workshop.avif` |
| Networking | `imgs/networking.jpg` |

PyBullet loads JPG/PNG textures reliably. If an AVIF decoder is unavailable in
the local Python image stack, those images fall back to the OpenCV preview window;
converting `food.avif` and `workshop.avif` to JPG or PNG gives the most reliable
tablet display.

## Layout

```
imgs/              # event images shown on Pepper's tablet
src/
  contracts.py      # shared Evidence/Result types + cross-package interfaces
  state_machine.py  # six-state interaction FSM (WP1)
  perception/       # face detection + camera sources (WP1)
  behaviour/        # Pepper speech, gestures, TTS, and tablet image display
  stubs.py          # console stand-ins for WP2/WP3/WP4
  main.py           # orchestrator / entry point
```
