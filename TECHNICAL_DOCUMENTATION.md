# JARVIS — Social Event Recommendation Agent on Pepper — Complete Technical Documentation

> **Agent name:** JARVIS · **Slogan:** *"At your service. Scanning the local grid
> for maximum energy signatures."*
>
> A socially interactive recommendation agent for the **Pepper** humanoid robot
> (running in the **qiBullet** simulation). It notices an approaching person,
> greets them with coordinated speech and gesture, holds a short natural-language
> conversation to learn their preferences, reasons over those preferences with a
> transparent three-layer **Bayesian network**, and recommends one of eight social
> events out loud — showing the matching image on Pepper's tablet and explaining
> *why* it chose it.
>
> This document is the exhaustive engineering reference for the system. It covers
> every module, class, method, algorithm, data structure, configuration knob,
> protocol message, dependency, and design decision. Nothing is summarised away.

---

## Table of Contents

1. [System overview & design philosophy](#1-system-overview--design-philosophy)
2. [High-level architecture (the single route)](#2-high-level-architecture-the-single-route)
3. [Repository layout (full file inventory)](#3-repository-layout-full-file-inventory)
4. [The two-runtime split and the environments](#4-the-two-runtime-split-and-the-environments)
5. [Shared contracts (the data currency)](#5-shared-contracts-the-data-currency)
6. [WP1 — Perception](#6-wp1--perception)
7. [WP1 — The six-state interaction FSM](#7-wp1--the-six-state-interaction-fsm)
8. [WP1/WP5 — The orchestrator (`main.py`)](#8-wp1wp5--the-orchestrator-mainpy)
9. [WP2 — The dialogue service (LangGraph + LLM)](#9-wp2--the-dialogue-service-langgraph--llm)
10. [WP3 — The Bayesian recommendation engine](#10-wp3--the-bayesian-recommendation-engine)
11. [WP4 — The behaviour / presentation layer](#11-wp4--the-behaviour--presentation-layer)
12. [WP5 — Integration: the process bridge](#12-wp5--integration-the-process-bridge)
13. [The stubs (development scaffolding)](#13-the-stubs-development-scaffolding)
14. [End-to-end data-flow walkthrough](#14-end-to-end-data-flow-walkthrough)
15. [Configuration reference (all environment variables & CLI flags)](#15-configuration-reference)
16. [Testing](#16-testing)
17. [Robustness & fallback matrix](#17-robustness--fallback-matrix)
18. [Dependency reference](#18-dependency-reference)
19. [Glossary](#19-glossary)

---

## 1. System overview & design philosophy

The agent combines three complementary ideas, each chosen for a specific reason:

1. **A finite-state interaction manager (FSM).** Gives the encounter a clear,
   robust shape (`Idle → Greeting → Conversation → Reasoning → Recommendation →
   Farewell → Idle`). Makes the visible robot behaviour predictable and easy to
   reset on interruption.
2. **A language model (LLM).** Turns casual, free-form replies into structured
   preferences and asks open, human-sounding questions. It is used *purely as an
   interface layer* — it never touches the model structure or the probabilities.
3. **A Bayesian network.** Produces an **explainable** recommendation rather than
   a black-box answer. Every observed preference maps to interpretable latent
   factors, and the recommendation can be read back as a "because…" sentence.

The overriding **design principle** (stated in `PLAN.md §1`): *the user speaks
naturally, but the final decision stays inspectable.* Inspectable means
explicit evidence values plus interpretable latent factors — never a learned
opaque scoring function.

A key structural consequence of this principle is the **LLM sandbox**: every
value the language model emits is passed through a **whitelist** before it can
reach the reasoning engine. The LLM can only ever emit legal labels; anything
else is silently dropped. This appears twice (once on each side of the process
boundary) and is the single most important safety invariant in the codebase.

### The six-state interaction cycle (canonical table)

| State | Robot action | Key transition out |
|---|---|---|
| **Idle** | Passive camera monitoring | Face *stable* for several consecutive frames |
| **Greeting** | Face verification (identify, or ask name + consent to enroll) + JARVIS greeting (wave → nod → open-arms) | User remains present |
| **Conversation** | Preference questions with clarification | Required evidence collected |
| **Reasoning** | "Let me think" thinking pose while inference runs | Posterior computed |
| **Recommendation** | Top events + explanation + open-palm gesture + tablet image | User acknowledges / timeout |
| **Farewell** | Goodbye speech + wave; reset | Return to Idle |

### Work packages (WPs)

The project is organised into five work packages, referenced throughout the code
in docstrings and comments:

- **WP1** — Perception & State Machine (the spine / orchestrator).
- **WP2** — Dialogue Manager + LLM Evidence Parser (LangGraph service).
- **WP3** — Bayesian Recommendation Engine (pyAgrum).
- **WP4** — Behaviour / Presentation Layer (speech + gestures + tablet).
- **WP5** — Integration, robustness & demo (the 3.8↔3.11 bridge, wiring).

---

## 2. High-level architecture (the single route)

The system is a **single route** (based on Kopp & Hassan, 2022, slides 7–8),
colour-coded in `architecture.pdf` as red = processing, green = decision,
blue = generation, grey = knowledge:

```
User (face + speech)
   │
   ▼
Perception ──► Interpretation ──► Dialogue Manager ──► Bayesian Reasoning ──► Behaviour & Acting ──► User
(sense+recog.)   (understand,LLM)   (manage turn,FSM)    (decide, pyAgrum)      (create+perform)      (sees+hears)
 Haar+5-frame     Groq Llama-3.3     LangGraph slot-       3-layer Bayes net      TTS + gesture +
 ASR:whisper      frames & parses    filling loop          6 ev→3 latent→8 ev     tablet image ·
 or typed text    → 6 slots          (expanded below)      ranked + explanation   Pepper/qiBullet
```

The **Dialogue Manager** stage is itself a LangGraph, expanded during the FSM's
`Conversation` state. It fills the six preference slots one at a time, with LLM
reframing and conflict handling:

```
START ─► Router ─► QuestionFramer ─► AnswerParser ─► ConflictResolver ─► Evaluator ─► (Router | Finish ─► END)
           ▲  "missing"    │ question out via         │ uses            │ conflict→ask either/or   │
           │               │ Behaviour-TTS,           │ Interpretation- │ (loops back to Framer)   │
           │               │ answer in via            │ LLM             │ answer not usable→reframe │
           └───────────────┴─ Perception-ASR ─────────┴─────────────────┴──────────────────────────┘
                                                        (usable / committed / default → next slot)
```

The route runs once per encounter: perception → interpretation → dialogue
(looping per turn) → Bayesian reasoning → behaviour → Pepper. **One
recommendation is reasoned once all six preferences have been gathered** (or the
conversation otherwise completes with partial evidence).

**Knowledge & data** (grey) feeds the reasoning and presentation: Bayesian CPTs
derived from the event catalogue, the 8 events + their images in `imgs/`, and the
schema in `contracts.py`.

---

## 3. Repository layout (full file inventory)

```
Project/
├── pyproject.toml                    # Robot side (Python 3.8) project + deps
├── uv.lock                           # Locked robot-side dependencies
├── .python-version                   # Pins the robot interpreter (3.8)
├── README.md                         # User-facing project readme
├── PLAN.md                           # Working plan: scope, WPs, interfaces, decisions
├── TECHNICAL_DOCUMENTATION.md        # (this file)
├── architecture.pdf / .pptx          # Single-route architecture diagram
├── Social-Event-...-Presentation*.pptx  # Slide decks
│
├── src/                              # ── ROBOT SIDE (Python 3.8) ──
│   ├── .env                          # GROQ/GROK key + LangSmith config (gitignored)
│   ├── main.py                       # WP1/WP5 orchestrator & CLI entry point
│   ├── state_machine.py              # WP1 six-state InteractionFSM
│   ├── contracts.py                  # Shared Evidence/Result types + Protocols + whitelist
│   ├── stubs.py                      # Console/scripted stand-ins for WP2/WP3/WP4
│   ├── dialogue_bridge.py            # WP5 robot-side client for the dialogue service
│   ├── check_perception.py           # Standalone perception diagnostic tool
│   ├── enroll_face.py                # Standalone CLI to pre-enroll a face offline
│   ├── perception/
│   │   ├── __init__.py               # Re-exports the perception classes
│   │   ├── face_detector.py          # WP1 sources + Haar detector + wrappers + scripted
│   │   ├── face_recognizer.py        # WP1 optional: LBPH face verification + enrollment
│   │   └── known_faces/              # Enrolled face photos (local-only, gitignored)
│   ├── behaviour/
│   │   ├── __init__.py               # Re-exports PepperBehaviour
│   │   ├── pepper_behaviour.py       # WP4 gestures + speech coordination
│   │   ├── tts.py                    # WP4 text-to-speech worker (edge-tts / pyttsx3)
│   │   ├── event_display.py          # WP4 tablet image display + fallbacks
│   │   └── tablet_panel.obj          # Mesh for the tablet overlay panel
│   └── recommender/
│       ├── __init__.py               # Re-exports BayesianRecommender, build_network
│       └── bayesian_network.py       # WP3 pyAgrum 3-layer network + recommender
│
├── dialogue/                         # ── DIALOGUE SERVICE (Python 3.11, own uv project) ──
│   ├── pyproject.toml                # Dialogue project + deps (LangGraph, openai, dotenv)
│   ├── uv.lock
│   ├── langgraph.json                # LangGraph Studio config (graph entry + env)
│   ├── README.md                     # WP2 readme
│   ├── RUNNING.md                    # How to run + view traces (LangSmith/Studio)
│   ├── .python-version               # Pins 3.11
│   ├── dialogue/                     # the package
│   │   ├── __init__.py
│   │   ├── schema.py                 # Evidence labels + whitelist (mirrors contracts.py)
│   │   ├── slots.py                  # Data-driven question plan (Slot dataclass)
│   │   ├── llm.py                    # Groq parser + Groq question framer + scripted parser
│   │   ├── inputs.py                 # Input providers (Typed/Speech/Scripted/Interrupt)
│   │   ├── graph.py                  # The LangGraph StateGraph (nodes + routing)
│   │   ├── manager.py                # DialogueManager.collect_evidence()
│   │   ├── asr.py                    # Local Whisper speech-to-text (faster-whisper)
│   │   ├── bridge.py                 # WP5 subprocess side (JSON-over-stdio)
│   │   ├── cli.py                    # Standalone terminal test harness
│   │   └── studio.py                 # LangGraph Studio entry point
│   └── tests/
│       └── test_graph_offline.py     # Offline graph tests (scripted parser + input)
│
├── tests/
│   └── test_recommender.py           # WP3 Bayesian recommender tests
│
├── imgs/                             # Event images shown on the tablet
│   ├── mueseum.jpg concert.jpg sports.jpg food.avif
│   └── outdoor.jpg nightlife.jpg workshop.avif networking.jpg
│
├── docs/
│   └── dialogue_graph.png            # Rendered dialogue graph
│
└── Proposal/                         # LaTeX proposal + Bayesian flowchart figures
```

---

## 4. The two-runtime split and the environments

### 4.1 Why two runtimes

The robot side (perception, behaviour, qiBullet/NAOqi bindings) is pinned to
**Python 3.8** for compatibility with NAOqi and qiBullet. LangGraph, the dialogue
framework, requires **Python ≥3.9** (the project uses **3.11**). A single
interpreter cannot satisfy both, so the two halves live in **separate uv
projects** with separate virtual environments and are connected by a subprocess
**bridge** (see §12).

### 4.2 Robot-side project — `Project/pyproject.toml`

- **Name:** `social-event-agent`, version `0.1.0`.
- **`requires-python = ">=3.8,<3.9"`** — hard-pinned to 3.8.
- **Dependencies:**
  - WP1 perception: `opencv-contrib-python>=4.5,<4.9` (the *contrib* build, so
    `cv2.face` is available for LBPH face verification/enrollment), `numpy<1.25`,
    `qibullet>=1.4.3`, `pybullet>=3.2.7`.
  - WP3 recommender: `pyagrum>=1.13.2`.
  - WP4 behaviour: `pyttsx3>=2.99` (offline TTS), `edge-tts>=6.1` (neural TTS),
    `playsound==1.2.2` (audio playback for neural TTS), `threadpool>=1.3.2`,
    `pillow-avif-plugin==1.4.6` (AVIF decode support for two of the event tablet
    images; pinned to the last version shipping a prebuilt cp38-win_amd64 wheel —
    newer versions need to compile against libavif headers that aren't installed).
- **Dev group:** `pytest>=8.0`.
- **`[tool.uv] package = false`** — standalone project, not a workspace member.
- **Pytest config:** `pythonpath = ["src"]`, `testpaths = ["tests"]`.

### 4.3 Dialogue-side project — `Project/dialogue/pyproject.toml`

- **Name:** `dialogue-service`, version `0.1.0`.
- **`requires-python = ">=3.11"`**.
- **Core dependencies:** `langgraph>=0.2.0`, `openai>=1.40` (used to talk to Groq
  through its OpenAI-compatible API), `python-dotenv>=1.0`.
- **Optional extra `speech`** (installed with `uv sync --extra speech`, kept
  separate so a `pyaudio` build failure can't block the core graph):
  `SpeechRecognition>=3.10`, `pyaudio>=0.2.13`, `faster-whisper>=1.0` (local ASR),
  `nvidia-cublas-cu12>=12.1`, `nvidia-cudnn-cu12>=9.1,<10` (CUDA 12 runtime for
  GPU Whisper; harmless on CPU-only machines — the ASR registers these DLL dirs at
  import time).
- **Dev group:** `langgraph-cli[inmem]>=0.4.30`, `pytest>=8.0`.
- **`[tool.uv] package = false`**; pytest `pythonpath = ["."]`, `testpaths =
  ["tests"]`.

### 4.4 Environment management

Both sides are managed with **uv**. Setup:

```bash
# robot side (Python 3.8)
uv sync

# dialogue side (Python 3.11)
cd dialogue
uv sync                 # core graph + LLM
uv sync --extra speech  # add spoken input (local Whisper)
```

### 4.5 Secrets & tracing — `src/.env`

The dialogue LLM needs a free **Groq** API key. It, plus optional LangSmith
tracing config, lives in `Project/src/.env` (gitignored). Both projects load it:
the dialogue side reads `../src/.env` (and a local `dialogue/.env` overrides it).

```ini
GROQ_API_KEY="gsk_..."           # free key from console.groq.com (or GROK_KEY)
GROQ_MODEL="llama-3.3-70b-versatile"   # optional model override

# optional — trace the dialogue graph at smith.langchain.com
LANGSMITH_API_KEY="lsv2_..."     # MUST be this exact variable name
LANGSMITH_TRACING="true"
LANGSMITH_PROJECT="dialogue"
```

> Gotcha documented in `RUNNING.md`: the tracing key **must** be named
> `LANGSMITH_API_KEY`; any other name is ignored and nothing appears in LangSmith.

---

## 5. Shared contracts (the data currency)

Two files define the data that flows between packages. They are **intentionally
duplicated** across the two runtimes because the interpreters cannot share code —
the contract between them is the *JSON shape* of the evidence dict.

### 5.1 `src/contracts.py` (robot side, 3.8-compatible)

This module is the single source of truth on the robot side. It is deliberately
dependency-free and Python-3.8 compatible so both sides can import it in principle.

**`AGENT_NAME = "JARVIS"`** and **`AGENT_SLOGAN = "At your service. Scanning the
local grid for maximum energy signatures."`** — the agent's identity, defined
once here and imported everywhere it's spoken or printed (`state_machine.py`'s
greeting/farewell lines, `main.py`'s startup banner).

**`EVIDENCE_LABELS`** — legal label values for each of the six preference slots.
Any slot may be *missing* from a concrete evidence dict (partial evidence is
explicitly allowed):

| Slot | Legal values |
|---|---|
| `Budget` | `Low`, `Med`, `High` |
| `GroupSize` | `Solo`, `Small`, `Large` |
| `ActivityLevel` | `Relaxed`, `Moderate`, `Active` |
| `Setting` | `Indoor`, `Outdoor`, `Either` |
| `TimeOfDay` | `Day`, `Evening`, `Night` |
| `Interest` | `Arts`, `Music`, `Food`, `Sports` |

- **`EVIDENCE_SLOTS`** — `tuple(EVIDENCE_LABELS.keys())`, the canonical question
  order and default slot-asking priority.
- **`EVENT_LABELS`** — the eight event categories the recommender ranks over
  (output layer): `Museum`, `Concert`, `Sports`, `Food`, `Outdoor`, `Nightlife`,
  `Workshop`, `Networking`.
- **Type aliases** (documentation only, not enforced): `Evidence = Dict[str,
  str]`; `Result = List[Dict[str, object]]` (e.g. `[{"event": "Museum", "prob":
  0.31}, ...]`, sorted descending).

**Functions:**

- `is_valid_evidence_value(slot, value) -> bool` — true iff `value` is a legal
  label for `slot`.
- `validate_evidence(evidence) -> Evidence` — returns a **clean copy** containing
  only legal slot/value pairs. Unknown slots and illegal values are dropped. This
  is the defensive guard so an LLM can never inject values the Bayesian network
  does not understand. Handles `None` gracefully (`evidence or {}`).
- `missing_slots(evidence) -> List[str]` — the slots (in canonical order) not yet
  filled with a legal value; computed after validation.

**Protocols** (structural typing via `typing.Protocol`, `@runtime_checkable`, so
stubs need no inheritance):

- `Perception` — `tick() -> bool`, `face_stable() -> bool`, `is_present() ->
  bool`, `reset() -> None`, `release() -> None`.
- `DialogueManager` — `collect_evidence() -> Evidence`.
- `Recommender` — `recommend(evidence) -> Result`, `explain(event, evidence) ->
  str`.
- `Behaviour` — `say(text)`, `gesture(name)`, `say_with_gesture(text, name)`
  (start both simultaneously, block until *both* finish), `present(result)`.

### 5.2 `dialogue/dialogue/schema.py` (dialogue side, 3.11)

A **mirror** of the evidence half of `contracts.py`, kept in sync manually. It
holds the same `EVIDENCE_LABELS` table, `EVIDENCE_SLOTS`, and `Evidence` alias,
plus:

- `is_valid(slot, value) -> bool` — same semantics as `is_valid_evidence_value`.
- `whitelist(candidate) -> Evidence` — keeps only legal slot/value pairs. This is
  the dialogue-side equivalent of `validate_evidence`, and it is applied to
  **every** LLM output before it leaves the parser.

> The docstring explicitly notes the duplication is intentional and that the two
> files MUST stay in sync.

---

## 6. WP1 — Perception

File: `src/perception/face_detector.py` (+ `check_perception.py` diagnostic).

OpenCV (`cv2`) is imported **lazily** at module top with a `try/except`, so the
scripted (camera-free) perception works in environments where `cv2` is absent.

### 6.1 Frame sources

**`WebcamSource(index=0)`** — a local webcam via `cv2.VideoCapture`.
- Requires `cv2`; raises `RuntimeError` otherwise.
- **Windows reliability fix:** the default MSMF backend often opens the device but
  then fails to grab frames (`CvCapture_MSMF::grabFrame ... -1072875772`). The
  `_open` helper therefore tries backends in order: on Windows (`os.name ==
  "nt"`) it prefers **DirectShow** (`cv2.CAP_DSHOW`), then falls back to
  `cv2.CAP_ANY`. For each backend it checks `isOpened()` **and** actually reads
  one frame (because MSMF can report `isOpened()` yet fail to stream); the first
  backend that delivers a frame wins.
- `read()` returns the BGR frame or `None`; `release()` frees the capture.

**`PepperCameraSource(pepper, camera_id=None, fps=15.0)`** — frames from Pepper's
simulated camera in qiBullet.
- Imports `Camera, PepperVirtual` from `qibullet` lazily (module loads without
  qibullet installed).
- Defaults to `PepperVirtual.ID_CAMERA_TOP`; subscribes at `Camera.K_QVGA`
  resolution.
- `read()` calls `pepper.getCameraFrame(handle)` (BGR, OpenCV-compatible);
  `release()` unsubscribes.

### 6.2 `HaarFaceDetector` — the core detector with a smoothing window

Implements the `Perception` contract. Constructor parameters:

| Param | Default | Meaning |
|---|---|---|
| `source` | — | any object with `read()` |
| `window` | `5` | length of the sliding boolean window (deque `maxlen`) |
| `stable_threshold` | `4` | frames-with-face needed (window full) to be *stable* |
| `present_threshold` | `2` | frames-with-face needed to still be *present* |
| `cascade_path` | `None` | defaults to OpenCV's bundled `haarcascade_frontalface_default.xml` |
| `min_face_size` | `60` | minimum face box size in px |
| `recognizer` | `None` | optional `FaceRecognizer` (§6.7) — enables face verification |

- Loads the Haar cascade from `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`
  and raises if the classifier is empty.
- **`tick()`** — reads one frame, converts to grayscale, runs
  `detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))`.
  Appends a boolean (`len(faces) > 0`) to the window and returns it. Stores
  `last_frame` and `last_faces` for diagnostics/overlays. **If a `recognizer` was
  passed**, also picks the largest detected box (`max(faces, key=lambda b: b[2]*b[3])`)
  and calls `recognizer.identify(gray, box)`, caching the result in
  `self.last_identity` (`Optional[str]`).
- **`identify()`** — returns `self.last_identity` (the WP1 optional face
  verification result for the current frame, or `None` if unrecognized/no
  recognizer wired).
- **`face_stable()`** — the **Idle→Greeting trigger**: true only when the window
  is *full* (`len == maxlen`) **and** at least `stable_threshold` (4) of those 5
  frames contained a face. This strictness avoids false triggers.
- **`is_present()`** — a *looser* bar (≥ `present_threshold` = 2 of the window):
  brief missed detections during the interaction don't look like the user left.
- **`reset()`** — clears the window (used on return to Idle).
- **`release()`** — releases the underlying source.

The two-threshold design is the smoothing mechanism referenced throughout: a face
must *persist* to start an interaction, but a momentary dropout won't end one.

### 6.3 `PreviewPerception` — transparent preview decorator

Wraps any detector exposing `last_frame`/`last_faces` and renders a live OpenCV
window on every `tick()`, drawing green rectangles around faces and a status label
whose colour encodes state:

- **STABLE** → green `(0,255,0)`
- **present** → amber `(0,200,255)`
- **searching** → red `(0,0,255)`

It forwards the full `Perception` interface, so it can drop in anywhere a detector
is expected. The window updates whenever the FSM ticks perception. It also
forwards `identify()` and exposes `last_frame`/`last_faces`/`recognizer` as
pass-through properties, so the FSM's enrollment flow (§7.4) can reach the inner
detector's live frame/recognizer through the decorator transparently.

### 6.4 `ThreadedPerception` — background capture + preview

The problem it solves: the OpenCV preview window only refreshes when the FSM calls
`tick()`, so it appears to "freeze" the moment the interaction blocks (e.g.
waiting on console input during dialogue, or sleeping through a gesture).

`ThreadedPerception(inner, preview=True, fps=15.0, window_name=...)` runs the
inner detector on its **own daemon thread**:

- The thread loops at `1/fps` (period), each cycle: acquire lock → `inner.tick()`
  → cache `detected`, `last_frame`, `last_faces`, `face_stable()`, `is_present()`.
  All OpenCV highgui calls happen on this single background thread (required for
  stability).
- **FSM-facing methods are cheap, lock-guarded accessors:** `tick()` just returns
  the latest cached `_detected`; `face_stable()`/`is_present()`/`reset()` delegate
  under the lock. `identify()` and the `last_frame`/`last_faces`/`recognizer`
  properties are likewise lock-guarded pass-throughs to the inner detector, so the
  enrollment flow (§7.4) can read the live frame/recognizer from whichever thread
  owns the FSM without racing the capture thread.
- `release()` sets the stop event, joins the thread (2 s timeout), releases the
  inner source, and destroys the window.

### 6.5 `ScriptedPerception` — camera-free stand-in

Deterministic `Perception` driven by an internal tick counter — used for tests
and the headless demo (no qiBullet, no webcam).

- `ScriptedPerception(stable_after=3, leaves_after=None)`.
- `tick()` increments `_t`; the user is "present" once `_t >= stable_after`, and
  (if `leaves_after` is set) "leaves" once `_t >= leaves_after`.
- `face_stable()` and `is_present()` both return the current presence latch.
- `reset()` clears only the presence latch but **keeps** the tick counter, so a
  scripted "leave" stays consistent across an Idle reset.

In `main.py` the scripted source uses `stable_after=3, leaves_after=12` so the
face appears quickly and the user "leaves" shortly after the recommendation,
letting the acknowledgement wait return promptly and the demo end cleanly.

### 6.6 `check_perception.py` — standalone diagnostic

A CLI tool to confirm detection works *before* wiring it into the FSM. Shows a
live window with detection boxes and the smoothing status (`detected` / `present`
/ `STABLE`), or (`--no-window`) prints a status line per frame (headless/SSH).

- `--source {webcam,pepper}`, `--camera-index N`, `--headless`, `--no-window`.
- For `pepper` it launches a `SimulationManager`, spawns Pepper, and returns a
  `PepperCameraSource` + a cleanup that stops the sim.
- Loop: `det.tick()`, read `is_present()`/`face_stable()`, draw boxes + coloured
  status text, quit on `q`/Esc. Prints an explicit error if `cv2` is missing.

### 6.7 `face_recognizer.py` — face verification & consent-based enrollment (optional #1/#2)

File: `src/perception/face_recognizer.py`. On-device face **recognition** (not
just detection) using OpenCV's `cv2.face` module — LBPH (Local Binary Patterns
Histograms), available only in the *contrib* build (`opencv-contrib-python`, see
§4.2). Imported lazily so the rest of perception still works if `cv2.face` is
absent (e.g. a non-contrib OpenCV install).

- **`FACE_SIZE = (200, 200)`** — the fixed size every enrolled/query face crop is
  normalised to before training/inference.
- **`_default_known_dir()`** — `<perception module dir>/known_faces`, the root of
  one subdirectory per enrolled person.
- **`preprocess(gray_frame, box) -> ndarray`** — crops the detector's `(x, y, w,
  h)` box out of a grayscale frame and resizes it to `FACE_SIZE`. Shared by the
  recognizer, `enroll_face.py`, and the FSM's live-enrollment path so training and
  inference always see identically shaped input.
- **`FaceRecognizer(known_dir=None, confidence_threshold=75.0)`:**
  - `_enrolled_people()` — lists the subdirectories of `known_dir` (one per name).
  - `_train()` — loads every sample image under each person's directory, builds a
    `cv2.face.LBPHFaceRecognizer_create()`, and trains it. Sets `self.available =
    True` on success; called once at construction, and again by `retrain()`.
  - **`identify(gray_frame, box) -> Optional[str]`** — preprocesses the box,
    predicts `(label, confidence)`. LBPH confidence is a **distance**, so lower is
    a better match: returns the matched name only if `confidence <=
    confidence_threshold` (default `75.0`), else `None`. Empirically a genuine
    match scores close to `0.0`; a stranger scores roughly `180`, so the default
    threshold sits with wide margin on both sides.
  - **`enroll_sample(frame, box, name) -> str`** — converts BGR→gray if needed,
    preprocesses, and saves the crop to `known_faces/<name>/NNN.jpg` (auto
    zero-padded index). Returns the saved path.
  - **`retrain()`** — re-runs `_train()` so a freshly enrolled person is
    recognizable immediately, without restarting the process.

`src/perception/known_faces/` is **git-ignored** — enrolled photos are biometric
data and stay local-only, never committed.

### 6.8 `enroll_face.py` — standalone offline enrollment CLI

File: `src/enroll_face.py`. Pre-enrolls a person outside of a live interaction —
useful for setting up known users ahead of a demo. Opens the webcam via the same
`HaarFaceDetector` + `WebcamSource` used at runtime (so detection behaves
identically), then every `--interval` seconds saves the largest detected face
(via `preprocess()`) to `perception/known_faces/<name>/`.

- **Flags:** `--name` (required), `--count` (default `20`), `--camera-index`
  (default `0`), `--interval` (seconds between captures, default `0.3`).
- Shows a live preview window with the detection box and a `saved/count` counter;
  `q`/Esc stops early. Exits with an explicit error if `cv2` isn't installed.
- Example: `python enroll_face.py --name Sunesh --count 30 --camera-index 1`.

This CLI and the FSM's **live, consent-based enrollment** (§7.4/§7.5) write to the
exact same directory layout, so either path — pre-enrolling offline or being
asked live by JARVIS the first time it sees someone — produces samples the other
can build on.

---

## 7. WP1 — The six-state interaction FSM

File: `src/state_machine.py`. Class `InteractionFSM`.

The FSM owns the control loop and calls into the other packages *only* through the
`contracts` interfaces, so any of them can be a stub. Robustness (proposal §6) is
handled here: if the user disappears mid-interaction the FSM resets to Idle, and a
weak/empty recommendation still presents whatever is available.

### 7.1 The `State` enum

`IDLE`, `GREETING`, `CONVERSATION`, `REASONING`, `RECOMMENDATION`, `FAREWELL`
(values are the human-readable names).

### 7.2 Constructor

```python
InteractionFSM(perception, dialogue, recommender, behaviour,
               idle_poll=0.1, presence_check_frames=3, ack_timeout=15.0,
               name_provider=None, get_text=None, enroll_samples=15)
```

| Param | Default | Meaning |
|---|---|---|
| `perception` | — | WP1 `Perception` |
| `dialogue` | — | WP2 `DialogueManager` |
| `recommender` | — | WP3 `Recommender` |
| `behaviour` | — | WP4 `Behaviour` |
| `idle_poll` | `0.1` s | sleep between perception polls |
| `presence_check_frames` | `3` | frames sampled per presence check |
| `ack_timeout` | `15.0` s | how long to wait for user acknowledgement |
| `name_provider` | `None` | optional `() -> Optional[str]` for a personalised greeting when face verification (below) yields nothing |
| `get_text` | `None` | optional `() -> str` for typed replies during the enrollment name/consent exchange; defaults to `_default_get_text` (console `input()`) |
| `enroll_samples` | `15` | number of live camera samples to capture when enrolling a new face |

Internal state: `state = State.IDLE`, `evidence = {}`, `result = []`,
`_running = False`.

### 7.3 The run loop

`run(max_cycles=None)` dispatches to a per-state handler dict. Each handler
returns the next `State`. When a `FAREWELL → IDLE` transition completes, one cycle
is counted; the loop stops after `max_cycles` completed interactions (or runs
forever if `None`). `KeyboardInterrupt` is caught for a clean shutdown; `_running`
is reset in a `finally`. `stop()` clears `_running`.

### 7.4 Helpers

- **`_still_present()`** — samples `presence_check_frames` frames (each
  `perception.tick()` then `sleep(idle_poll)`), then returns `is_present()`. Used
  to detect the user walking away between phases.
- **`_reset_interaction()`** — best-effort `behaviour.clear_display()` (via
  `getattr`, tolerant of stubs without it), then clears `evidence`, `result`, and
  calls `perception.reset()`.
- **`_default_get_text()`** — `staticmethod`; the console fallback for
  `get_text`: `input("  [you] ").strip()`, returning `""` on `EOFError`.
- **`_capture_enrollment_samples(recognizer, name) -> int`** (optional #2) —
  captures live samples of the person currently in view for a fresh enrollment,
  **reusing the FSM's already-open camera feed** (`self.perception`) rather than
  opening a second competing webcam connection. Loops up to `enroll_samples * 4`
  attempts: each iteration calls `perception.tick()`, reads `last_frame` /
  `last_faces` (via `getattr`, tolerant of perception objects without them — e.g.
  `ScriptedPerception`), and if a face is present picks the largest box
  (`max(faces, key=lambda b: b[2]*b[3])`) and calls
  `recognizer.enroll_sample(frame, box, name)`, incrementing `saved` on success
  (individual capture failures are logged at debug level and skipped, not fatal).
  Sleeps `0.15` s between attempts. Returns the number of samples actually saved.
- **`_identify_or_enroll() -> Optional[str]`** (optional #1/#2) — the face
  verification + consent-based enrollment flow, called at the start of
  `_greeting()`:
  1. If `perception.identify` exists and returns a name, that name is returned
     immediately (known, authorized visitor — no camera-facing questions needed).
  2. Else, if `perception.recognizer` is `None` (no camera source wired for this
     run, e.g. `--source scripted`), returns `None` — nothing to enroll into.
  3. Else (webcam/Pepper source, unrecognized face): asks *"I don't think we've
     met yet. What's your name?"*, reads a name via `get_text()`. An empty name
     aborts (returns `None`).
  4. Asks *"Nice to meet you, {name}. Would it be okay if I remember your face
     for next time?"* and reads consent via `get_text()`. If the reply starts
     with `"y"` (case-insensitive): captures `enroll_samples` live samples
     (`_capture_enrollment_samples`), and if any were saved, calls
     `recognizer.retrain()` and confirms *"Great, I'll remember you next time,
     {name}!"*; if none were captured, apologises and continues without enrolling.
     If consent is declined, says *"No problem, {name}."* and captures nothing.
     Either way the given name is returned and used for this conversation.

### 7.5 State handlers

- **`_idle()`** — passively `tick()`s until `face_stable()` is true (→ Greeting).
  At that transition it also calls `perception.identify()` (guarded via
  `getattr`, so stubs without it don't break) purely to **log** whether the
  stable face is an authorized/known visitor or unrecognized — the actual
  verify-or-enroll decision happens in `_greeting()`. Sleeps `idle_poll` between
  polls.
- **`_greeting()`** — first calls `_identify_or_enroll()` (§7.4 — optional
  #1/#2: verifies the face, or asks a new visitor's name and, with consent,
  enrolls it live). If that returns `None` (no camera source wired, or the
  visitor declined to give a name), falls back to `name_provider()` if one was
  given. Builds the greeting line:
  `"Greetings, {name}. {AGENT_NAME} online, at your service."` if a name was
  resolved, else `"Greetings. {AGENT_NAME} online, at your service."` (`AGENT_NAME`
  = `"JARVIS"`, from `contracts.py`, §5.1). Then **three coordinated beats**, each
  a `say_with_gesture` that starts the line and gesture together and blocks until
  both finish:
  1. the greeting line above + `wave`
  2. `"Systems nominal. It's good to see you."` + `nod`
  3. `"Scanning the local grid for a social event to suit you."` + `open_arms`
  Then `_still_present()`; if the user left → Idle, else → Conversation.
- **`_conversation()`** — `raw = dialogue.collect_evidence()`, then
  `self.evidence = validate_evidence(raw)` (**the whitelist is applied again here**
  on the robot side, defence in depth). If the user left → Idle, else → Reasoning.
  (When running with the real dialogue bridge, each question is spoken with a
  brief, question-length `"talk"` gesture rather than a fixed-length one — see
  §8.4 — and abusive replies are declined by the dialogue graph itself, §9.5.3,
  before they ever reach evidence parsing.)
- **`_reasoning()`** — `say_with_gesture("Let me think about that.", "think")`,
  then `self.result = recommender.recommend(self.evidence) or []`. → Recommendation.
- **`_recommendation()`**:
  - If `result` is empty (robustness): says *"I'm sorry, I couldn't find a good
    match right now."* → Farewell.
  - Otherwise: `say_with_gesture("Based on what you told me, I recommend the
    following.", "present")`; then `behaviour.present(self.result)` (shows the top
    events + tablet image); then `recommender.explain(top_event, self.evidence)`
    (guarded — any exception → empty reason) and speaks it if non-empty; then
    `_wait_for_ack()`. → Farewell.
- **`_farewell()`** — `say_with_gesture("Enjoy your evening. {AGENT_NAME} signing
  off, goodbye.", "wave")`, then `_reset_interaction()`. → Idle.

### 7.6 Acknowledgement wait

`_wait_for_ack()` loops until `time.time()` passes `deadline = now + ack_timeout`
or the user is no longer present (`is_present()` false → "acknowledged/left"),
ticking perception and sleeping `idle_poll` each iteration.

---

## 8. WP1/WP5 — The orchestrator (`main.py`)

File: `src/main.py`. Constructs the components from CLI args, wires the FSM, and
runs it. Puts `src/` on `sys.path` so `python main.py` works from anywhere.

### 8.1 CLI arguments

| Flag | Choices / type | Default | Meaning |
|---|---|---|---|
| `--source` | `scripted`, `webcam`, `pepper`, `hybrid` | `scripted` | perception source / whether Pepper is spawned |
| `--camera-index` | int | `0` | webcam index |
| `--preview` | flag | off | show webcam preview window with detection boxes (auto-on for hybrid) |
| `--headless` | flag | off | qiBullet without GUI |
| `--interactive` | flag | off | ask preference questions on the console (stub dialogue) instead of defaults |
| `--dialogue` | `stub`, `real` | `stub` | console stand-in vs. the real LangGraph service via the bridge |
| `--answer` | `text`, `speech` | `text` | with `--dialogue real`: typed vs. spoken (local Whisper) answers |
| `--max-cycles` | int | `None` | stop after N interactions (defaults to 1 for scripted, unlimited otherwise) |
| `--verbose` | flag | off | debug logging |

### 8.2 `_spawn_pepper(args)`

Launches qiBullet: `SimulationManager().launchSimulation(gui=not headless)`,
`spawnPepper(client, spawn_ground_plane=True)`, `goToPosture("Stand", 0.6)`. If a
GUI is shown, frames the debug visualiser camera on Pepper's face
(`cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0,0,1.4]`).
Returns `(sim, client, pepper)`.

### 8.3 `build_components(args)` — perception + behaviour + cleanup per source

- **`scripted`** — `ScriptedPerception(stable_after=3, leaves_after=12)` +
  `ConsoleBehaviour()` + no-op cleanup. No camera, so no `FaceRecognizer` is
  built — face verification/enrollment (§6.7) is simply unavailable in this mode.
- **`webcam`** — builds a `FaceRecognizer()` (optional #1: verifies against
  authorized users enrolled under `known_faces/`; fails soft — with nobody
  enrolled yet it's a no-op and every face is just "unrecognized") and passes it
  to `HaarFaceDetector(WebcamSource(index), recognizer=FaceRecognizer())`;
  wrapped in `ThreadedPerception(preview=True)` iff `--preview`; +
  `ConsoleBehaviour()`; cleanup = `perception.release`.
- **`pepper`** — spawns Pepper, `PepperBehaviour(pepper)`, one shared
  `recognizer = FaceRecognizer()`, perception =
  `HaarFaceDetector(PepperCameraSource(pepper), recognizer=recognizer)`. Cleanup
  shuts down behaviour, releases perception, stops the sim.
- **`hybrid`** — Pepper's body in the sim (`PepperBehaviour`) **plus** webcam face
  detection via `ThreadedPerception(HaarFaceDetector(WebcamSource,
  recognizer=recognizer), preview=True)`, sharing the same `FaceRecognizer`
  instance built for the `pepper` branch. Same cleanup as `pepper`.

### 8.4 `_build_real_dialogue(behaviour, base_cleanup, args)` (WP5 wiring)

Wires the LangGraph dialogue service in via the bridge. Locates the `dialogue/`
directory (sibling of `src/`), and defines:

- `speak(question)` — prints `[Pepper asks]` and calls
  `behaviour.say_with_gesture(question, "talk")`. Requirement #6 (conversation
  gestures): this uses the brief, single-beat `"talk"` gesture (§11.2) rather
  than the longer `open_arms` swing, because `say_with_gesture` blocks the
  gesture for exactly as long as the line takes to speak — a long, fixed-length
  gesture visibly drifted out of sync with these variable-length, LLM-generated
  questions, whereas the short beat stays in sync regardless of question length.
- `get_text()` — reads a typed answer from the console (`input("  [you] ")`),
  returns `""` on EOF.
- `on_heard(text)` — prints `[you said] ...` (used in speech mode).
- `notify(message)` — prints `[Pepper notifies]` and calls
  `behaviour.say_with_gesture(message, "nod")` (status/clarification lines, e.g.
  the Evaluator's default-applied notice or the abuse-filter's "let's keep this
  friendly" prompt, get a small nod rather than no gesture at all).

Then constructs `DialogueBridge(speak, dialogue_dir, answer_mode=args.answer,
get_text=get_text, notify=notify, on_heard=on_heard)`. The returned cleanup
closes the bridge then runs the base cleanup. **If the bridge can't start** (any
exception), it logs a warning and falls back to `StubDialogueManager(interactive=...)`.

### 8.5 `main()`

1. Parse args, configure logging (`DEBUG` if `--verbose`, else `INFO`).
2. `build_components(args)`.
3. Dialogue: default `StubDialogueManager(interactive=...)`; if `--dialogue real`,
   swap for `_build_real_dialogue(...)` (which may itself fall back to the stub).
4. Recommender: try `BayesianRecommender()`; **on any exception** (pyAgrum or
   network unavailable) log a warning and use `StubRecommender()`.
5. FSM construction: for `scripted`, use fast polling and a short ack wait
   (`idle_poll=0.05, ack_timeout=2.0`) and default `max_cycles=1`; otherwise
   default construction and unlimited cycles.
6. Prints the startup banner: `"=== {AGENT_NAME} - {AGENT_SLOGAN} (source={args.source})
   ==="` (both from `contracts.py`, §5.1).
7. `fsm.run(max_cycles=...)` inside a `try/finally` that always runs `cleanup()`.

Example invocations (from the README):

```bash
uv run python src/main.py --source scripted                        # headless demo, one cycle
uv run python src/main.py --source webcam --interactive            # webcam + console dialogue
uv run python src/main.py --source pepper                          # Pepper in qiBullet
uv run python src/main.py --source scripted --dialogue real        # real LLM conversation
uv run python src/main.py --source hybrid --dialogue real --answer speech   # fully spoken & embodied
```

---

## 9. WP2 — The dialogue service (LangGraph + LLM)

Directory: `dialogue/dialogue/`. Runs as its own Python 3.11 process. Exposed to
the robot through one method: `DialogueManager.collect_evidence() -> Evidence`.

### 9.1 `schema.py`

Covered in §5.2 — the evidence labels + `is_valid` + `whitelist` guard, mirroring
`contracts.py`.

### 9.2 `slots.py` — the data-driven question plan

A single generic slot-filling loop reads this table, so adding/reordering
questions is just editing the list.

**`Slot`** (frozen dataclass): `name`, `question` (asked when empty), `clarify`
(re-asked when ambiguous), `default` (used after clarification attempts are
exhausted); plus a computed `options` property (`EVIDENCE_LABELS[name]`).

**`DEFAULT_SLOTS`** — the six slots in order, each with its canned question,
clarify wording, and default:

| Slot | Canned question | Default |
|---|---|---|
| `Budget` | "What's your budget like — low, medium, or high?" | `Med` |
| `GroupSize` | "Are you going solo, with a small group, or a large group?" | `Small` |
| `ActivityLevel` | "Do you feel like something relaxed, moderate, or active?" | `Moderate` |
| `Setting` | "Indoor, outdoor, or no preference?" | `Either` |
| `TimeOfDay` | "When are you thinking — daytime, evening, or night?" | `Evening` |
| `Interest` | "What are you into — arts, music, food, or sports?" | `Music` |

**`SLOT_HINTS`** — short natural-language hints given to the LLM question framer
so it can ask about each field meaningfully without reading the option list aloud
(e.g. `Budget` → "how much they'd like to spend (cheap / mid-range / pricey)").

### 9.3 `llm.py` — the interpretation layer

Constants: `GROQ_BASE_URL = "https://api.groq.com/openai/v1"`,
`DEFAULT_MODEL = "llama-3.3-70b-versatile"`.

#### 9.3.1 `GroqEvidenceParser` — free text → discrete evidence

- Constructor: resolves the API key from `GROQ_API_KEY` **or** `GROK_KEY` (raises
  if neither is set); model from `GROQ_MODEL` or the default; builds an `OpenAI`
  client pointed at the Groq base URL.
- **`parse(user_text, focus_slot=None, question=None) -> dict`** — builds a
  **system prompt** instructing the model to convert a casual reply into discrete
  values, returning a JSON object whose keys are a subset of the slots and whose
  values are exactly one of the allowed options, omitting anything unclear. The
  prompt encodes explicit **rules**:
  - Do **not** fill a slot from a vague/agreeable reply ("great", "cool", "ok",
    "sure", "yeah", "sounds good", "nice") — omit it.
  - Use a catch-all (`Setting='Either'`, `ActivityLevel='Moderate'`) **only** when
    the user explicitly says they have no preference, never as a fallback.
  - Greetings/pleasantries ("good day/morning/evening/night", "hello", "hi",
    "sir", "maam") are **not** preferences — never set `TimeOfDay` from a greeting.
  - Respect negation: "not hungry", "no sports", "not into music" mean the user
    does **not** want that.
  - Time mapping: tonight/late = Night; this evening/after work = Evening;
    today/this afternoon/morning = Day.
  - Group mapping: girlfriend/partner/a friend/a few people = Small; alone/by
    myself = Solo; big crowd/lots of people = Large.
  - The allowed values are appended via `_schema_description()`.
  - Context: if a `question`/`focus_slot` is provided they are woven into the user
    message ("The user was just asked: ...", "That question was mainly about ...").
  - Calls `chat.completions.create(..., response_format={"type":"json_object"},
    temperature=0)` — deterministic JSON mode. Parses the JSON; **any exception**
    (network/parse) logs a warning and returns `{}`.
  - **Every** return value is passed through `whitelist(...)` — the LLM sandbox.

#### 9.3.2 `GroqQuestionFramer` — conversational question generation

- Constructor: same key/model resolution; `temperature=0.7`.
- **`_STYLE_HINTS`** — a module-level tuple of six short style directives (e.g.
  *"Ask with calm, genuine curiosity."*, *"Ask using a brief, natural everyday
  comparison."*, *"Ask with understated, dry wit — one subtle touch, nothing
  more."*). One is picked at random (`random.choice`) on **every** `frame()` call
  and appended to the prompt as `"Style for this question: {style}"`. This is
  what keeps repeated turns — and repeated runs, since `Router` (§9.5.3) also
  randomizes which slot is asked next — from converging on the same phrasing;
  temperature alone tends to still favour one template for a near-identical
  prompt.
- **`frame(filled, missing, target, attempt=0, feedback=None) -> str | None`** —
  a system prompt casts the model as **"JARVIS, a composed, courteous
  social-event assistant running on a Pepper robot"**, told to ask ONE short
  (~20 words), open, experiential question to learn the `target`, weaving in
  what's already known, **without** reading out the options, no rigid
  this-or-that, no yes/no. The tone instruction is deliberately restrained:
  *"Keep the tone professional and understated — at most a subtle, dry touch of
  wit, used sparingly, never slang or forced jokes."* — plus *"Never reuse the
  same wording, structure, or example twice; genuinely vary your phrasing and
  angle every time."* Four few-shot examples set the bar (plain, polite,
  open-ended — e.g. *"Will it just be you this evening, or is anyone joining
  you?"*). The user message supplies `Already known: ...`, the `TARGET` + its
  `SLOT_HINTS` hint, optionally the `Other still-unknown` slots, optionally a
  `feedback` note (from the Evaluator reframe or the ConflictResolver either/or),
  and the chosen style hint. Uses `temperature=0.7, max_tokens=60`. On **any
  error** returns `None` so the graph falls back to the slot's canned wording.

#### 9.3.3 `ScriptedParser` — deterministic, offline keyword matcher

- Used for offline tests and as a graceful fallback when no LLM key is present.
- Holds a nested `KEYWORDS` dict: for each slot, each value maps to a list of
  keyword phrases. Full keyword lists:
  - **Budget**: Low = cheap/low/budget/inexpensive/affordable/free; Med =
    medium/moderate/mid/average/reasonable; High =
    expensive/high/premium/luxury/fancy/splurge.
  - **GroupSize**: Solo = solo/alone/myself/just me/by myself; Small = small/few
    friends/couple of/a few/small group; Large = large/big group/lots
    of/many/crowd/big.
  - **ActivityLevel**: Relaxed = relaxed/chill/calm/easy/laid back/quiet; Moderate
    = moderate/medium/balanced/normal; Active =
    active/energetic/sporty/intense/lively.
  - **Setting**: Indoor = indoor/inside/indoors; Outdoor =
    outdoor/outside/outdoors/open air/nature; Either = either/no
    preference/doesn't matter/any/whatever.
  - **TimeOfDay**: Day = day/daytime/morning/afternoon/noon; Evening =
    evening/dinner/sunset; Night = night/late/nightlife/midnight.
  - **Interest**: Arts = arts/art/museum/gallery/culture/painting; Music =
    music/concert/gig/band/live music; Food =
    food/eat/cuisine/restaurant/foodie/tasting; Sports =
    sports/sport/game/match/fitness/athletic.
- **`parse(...)`** — lowercases the text (padded with spaces), and for each slot
  takes the **first** value whose any keyword matches as a whole word (regex
  `\bkw\b`). The result is passed through `whitelist(...)`.

### 9.4 `inputs.py` — input providers

All implement `ask(question) -> str` and `notify(message) -> None`, so the graph
is agnostic to the source. This is the seam where the robot's ASR/TTS plug in.

- **`TypedInput`** — console: prints `[robot] <question>`, reads `input("[you ] ")`
  (returns `""` on EOF); `notify` prints `[robot] <message>`.
- **`InterruptInput`** — for LangGraph Studio/API: `ask` raises a LangGraph
  `interrupt({"question": question})` so the graph pauses and surfaces the
  question in Studio; the run resumes on `Command(resume="...")`. `notify` is a
  no-op (the next anchored question re-asks anyway). This is the same
  human-in-the-loop pattern the robot bridge uses.
- **`ScriptedInput(answers)`** — returns queued answers in order (then `""`); used
  by tests. `notify` is a no-op.
- **`SpeechInput(speak_questions=True, phrase_time_limit=8.0)`** — microphone STT
  via `SpeechRecognition`. Calibrates ambient noise on init; optionally speaks
  questions via `pyttsx3`. `ask` speaks the question, listens, and calls
  `recognizer.recognize_google(audio)` (the **Google** backend — note this is
  distinct from the local-Whisper path used in the WP5 bridge). Handles
  `UnknownValueError` ("didn't catch that" → `""`) and other errors gracefully.

### 9.5 `graph.py` — the LangGraph StateGraph

This is the heart of WP2. It defines the state schema, five agent nodes plus a
Finish node, and the conditional routing.

#### 9.5.1 `DialogueState` (a `typing_extensions.TypedDict`)

> Uses `typing_extensions.TypedDict` (not `typing.TypedDict`) because on
> Python < 3.12 pydantic — used by the LangGraph API/Studio to derive schemas —
> requires it.

| Field | Type | Meaning |
|---|---|---|
| `evidence` | dict | slots filled so far (the contract type) |
| `target` | Optional[str] | slot currently being resolved |
| `last_question` | str | the question just asked (parser context) |
| `last_user_text` | str | the latest reply |
| `candidate` | dict | cleaned parser output awaiting commit |
| `attempts` | int | reframe attempts on the current target |
| `feedback` | Optional[str] | note for the QuestionFramer to reframe/disambiguate |
| `conflict` | bool | set by ConflictResolver to route a disambiguation |
| `turn_log` | list | transcript (explainability + eval evidence) |
| `done` | bool | finished flag |

`initial_state()` returns all fields zeroed/empty.

#### 9.5.2 Weak-value guard

Module-level constants defend against the LLM guessing catch-all values:

- `_WEAK_VALUES = {"Setting": "Either", "ActivityLevel": "Moderate"}`.
- `_NO_PREFERENCE_CUES` — a tuple of phrases signalling genuine indifference:
  "either", "any", "anything", "no pref", "dont care", "don't care", "doesnt
  matter", "doesn't matter", "does not matter", "whatever", "up to you", "both",
  "flexible", "not fussed".
- `_has_no_preference_cue(text)` — true if any cue is a substring of the lowered
  text.

**Abusive-language guard (requirement #8, optional).** A second, independent
module-level constant defends the `AnswerParser` node against hostile input:

- `_ABUSE_WORDS = ("idiot", "stupid", "moron", "shut up")` — a small, deliberately
  short flagged-word list (kept minimal by design, rather than a full moderation
  model or a second LLM checkpoint — see §9.5.3).
- `_is_abusive(text) -> bool` — true if any flagged word is a **substring** of the
  lowered reply (simple containment check, no NLP).

#### 9.5.3 `build_graph(parser, input_provider, framer=None, slots=None, max_attempts=2, randomize_order=True)`

Builds `by_name` and `order` from the slot list; `missing_of(evidence)` returns
the slots (in order) not yet in evidence.

**Nodes:**

- **`Router`** — picks the next unknown slot. If none missing → `{done: True,
  target: None}`. Else, `target = random.choice(missing) if randomize_order else
  missing[0]` → `{done: False, target, attempts: 0, feedback: None}` (resets the
  reframe state for the new slot). `randomize_order` defaults to `True` in
  production (`build_graph`/`DialogueManager`, §9.6) so the conversation doesn't
  always open with the same slot (e.g. always `Budget`) in the same fixed order
  every run; offline tests that hard-code answers against the canonical order
  pass `randomize_order=False` for determinism (§16).
- **`QuestionFramer`** — if a `framer` is present, tries `framer.frame(evidence,
  missing, target, attempt, feedback)` (guarded; on error → `None`); if no LLM
  question, falls back to `by_name[target].question`. Calls
  `input_provider.ask(question)` to get the reply. Appends a turn record
  `{target, q, a, feedback}` to `turn_log`. Returns `last_question`,
  `last_user_text`, `turn_log`.
- **`AnswerParser`** — **first**, requirement #8 (optional): if
  `_is_abusive(last_user_text)` (the flagged-word substring check above), the
  reply is **not parsed at all** — `input_provider.notify("Let's keep this
  friendly — could you rephrase that?")` fires and the node returns
  `{candidate: {}}`, so nothing abusive can ever reach evidence, and the
  Evaluator will simply reframe the same target on the next turn as if the
  answer had been unusable. Otherwise, `candidate = whitelist(parser.parse(
  last_user_text, focus_slot=target, question=last_question))`, then the **weak-
  value guard**: if `candidate[target]` equals the weak value for that target
  **and** the user gave no explicit no-preference cue, that entry is dropped (so
  a vague "great" doesn't become a fake `Either`/`Moderate`). Returns
  `{candidate}`.
- **`ConflictResolver`** — finds slots present in *both* `candidate` and
  `evidence` with **different** values. If none → `{conflict: False}`. Otherwise
  takes the first conflict `slot` (old vs new): commits any *non-conflicting* new
  info from the same reply, **re-opens** the contradicted slot (removes it from
  evidence), and writes a `feedback` string instructing the framer to ask a short,
  clear either/or question to confirm old vs new. Returns updated `evidence`,
  empty `candidate`, `target=slot`, the feedback, `attempts=0`, `conflict=True`.
- **`Evaluator`** — commits only **new** slots (never silently overwrites; the
  weak-value/conflict cases are handled above). If `target` is now in evidence
  (filled directly or via multi-slot fill) → clear feedback, reset attempts, done.
  Else increments `attempts`; if `attempts > max_attempts` → apply the slot's
  `default`, reset attempts, and `input_provider.notify("No worries, I'll go with
  <default> for now.")`. Else set a `feedback` note telling the framer to ask
  again a different, open, natural way (no option lists; may give one concrete
  example).
- **`Finish`** — `{done: True}`.

**Routing:**

- `START → Router`.
- `Router` → `Finish` if `done` else `QuestionFramer`.
- `QuestionFramer → AnswerParser → ConflictResolver`.
- `ConflictResolver` → `QuestionFramer` if `conflict` else `Evaluator`.
- `Evaluator` → `Router` if `target in evidence` else `QuestionFramer` (reframe).
- `Finish → END`.

Compiled with `g.compile()` (no checkpointer here — `langgraph dev` supplies one
in Studio).

#### 9.5.4 Multi-slot fills and partial evidence

Because the parser may return several slots from one reply (e.g. "somewhere cheap
outside with a few friends" → Budget=Low, Setting=Outdoor, GroupSize=Small), the
Evaluator commits all new slots at once, and the Router simply skips any slot
already filled — so the conversation naturally shortens.

### 9.6 `manager.py` — the WP2 entry point

`DialogueManager(parser, input_provider, framer=None, slots=None, max_clarify=2,
recursion_limit=100, randomize_order=True)` builds and compiles the graph
(passing `randomize_order` through to `build_graph`, §9.5.3). `collect_evidence()`
invokes the graph from `initial_state()` with a `{"recursion_limit":
recursion_limit}` config and returns `final["evidence"]`. `run_verbose()` returns
the full final state (evidence + `turn_log`) for the CLI/debugging.

> `recursion_limit` defaults to `100` rather than LangGraph's stock `25`:
> `randomize_order=True` means a run can, by chance, revisit a slot's
> reframe/default cycle several times across all six slots before finishing, and
> a too-low limit (e.g. `5`) can trip `GraphRecursionError` on an unlucky ordering
> even though the graph itself is making normal progress.

### 9.7 `asr.py` — local speech-to-text (WP5, on-device)

Runs entirely on-device (no cloud) using **faster-whisper** (CTranslate2 backend).
Records a mic utterance with SpeechRecognition endpointing (stops on silence),
then transcribes locally — GPU if available, else CPU.

- **`_register_cuda_dlls()`** (Windows only) — the `nvidia-*-cu12` pip wheels drop
  their DLLs under `site-packages/nvidia/<lib>/bin`, which isn't on the loader
  path. This finds the `nvidia` package, collects the `bin` dirs for `cublas`,
  `cudnn`, `cuda_nvrtc`, `cuda_runtime`, registers each via
  `os.add_dll_directory`, **and** prepends them to `PATH` (because CTranslate2
  loads cuBLAS/cuDNN via plain `LoadLibrary`, which searches PATH, not the
  add-dll set). Called once at import.
- **`_pick_device(device, compute)`** — if an explicit device is given, uses it
  (default compute `int8_float16` on cuda, `int8` on cpu). Otherwise tries
  `ctranslate2.get_cuda_device_count() > 0` → `("cuda", "int8_float16")`, else
  `("cpu", "int8")`. The rationale (documented): on a GPU, `int8_float16` lets
  `large-v3` fit a 6 GB card (~2.2 GB vs ~3.4 GB for float16) with negligible
  accuracy loss and leaves headroom for the qiBullet sim.
- **`WhisperTranscriber`** — resolves device/compute and a **device-aware default
  model** (`large-v3` on cuda, `small` on cpu; overridable via `WHISPER_MODEL`),
  loads `faster_whisper.WhisperModel`. `_transcribe(wav_bytes)` runs
  `transcribe(..., language="en", beam_size=1)` and joins the segment texts.
  `transcribe_wav` wraps it: if GPU compute fails at runtime (missing
  cuBLAS/cuDNN), it logs a warning, reloads on **CPU** (`int8`) once, and stays
  there.
- **`get_transcriber()`** — a process-wide singleton (model loading is expensive).
- **`record_utterance(phrase_time_limit=12.0, timeout=12.0)`** — captures one
  utterance as WAV bytes via `speech_recognition`. Honours `WHISPER_MIC_INDEX`
  (input device) and `WHISPER_ENERGY` (fixed energy threshold; otherwise
  calibrates from ambient noise for 0.4 s).
- **`listen_and_transcribe(phrase_time_limit=12.0)`** — records then transcribes;
  returns `""` on mic failure (timeout/no mic) or transcription failure.
- **`__main__`** — a standalone mic + Whisper check that lists input devices and
  transcribes one utterance.

### 9.8 `bridge.py` — the subprocess side (see §12 for the full protocol)

Loads the `.env` (shared `../src/.env`, then a local `dialogue/.env` overriding),
configures logging to **stderr** (stdout is reserved for protocol JSON), and:

- **`_build_llm()`** — if `GROQ_API_KEY`/`GROK_KEY` is set → `(GroqEvidenceParser,
  GroqQuestionFramer)`; otherwise `(ScriptedParser, None)`. So the service
  gracefully runs offline with keyword parsing if no key is present.
- **`RemoteInput`** — an input provider whose `ask` emits `{"event":"ask",
  "question":...}` and waits for the robot's reply: `{"answer": ...}` (typed) or
  `{"spoken": true}` (→ `_listen()`, which records the mic and transcribes locally
  via `asr.listen_and_transcribe`, emitting `listening` then `heard`). `notify`
  emits `{"event":"notify","message":...}`.
- **`main()`** — emits `{"event":"ready"}`, then a command loop: `quit` breaks;
  `collect` builds a fresh `DialogueManager(parser, RemoteInput(), framer)`, runs
  `collect_evidence()`, and emits `{"event":"evidence","evidence":...}` (or
  `{"event":"error","message":...}` on exception).

### 9.9 `cli.py` — standalone terminal harness

Loads the `.env`, builds a parser (`groq`/`scripted`), a framer (only when
`llm=groq` and framing enabled), and an input (`text`/`speech`/`scripted`). Runs
`manager.run_verbose()`, prints the collected evidence, lists any missing slots,
and (with `--verbose`) prints the Q/A transcript. `DEMO_ANSWERS` is a 4-line
scripted conversation whose first line deliberately fills several slots at once
("somewhere cheap outside with a few friends"). Flags: `--llm {groq,scripted}`,
`--input {text,speech,scripted}`, `--no-frame`, `--verbose`.

### 9.10 `studio.py` — LangGraph Studio entry point

Loads the `.env`, then builds a compiled `graph` for `langgraph dev` using
`InterruptInput()` (so each question pauses the run and surfaces in Studio). Uses
the Groq parser + framer if a key is present, else the offline `ScriptedParser`
(so the graph still loads and renders). `langgraph dev` provides the checkpointer.

### 9.11 `langgraph.json` — Studio config

```json
{
  "dependencies": ["."],
  "graphs": { "dialogue": "./dialogue/studio.py:graph" },
  "env": "../src/.env"
}
```

Points Studio at the compiled `graph`, includes the local package as a dependency,
and loads the shared `.env`.

### 9.12 Observability

Two mechanisms (see `RUNNING.md`): **LangSmith tracing** (every run logged when
`LANGSMITH_*` is set — expand the run tree to see each node's input/output state
and the exact Groq prompt/response/tokens/latency) and **LangGraph Studio** (the
live graph topology; drive runs in the browser via the interrupt/resume box).

---

## 10. WP3 — The Bayesian recommendation engine

File: `src/recommender/bayesian_network.py`. Built with **pyAgrum**. A small,
explainable **three-layer** Bayesian network.

### 10.1 Network topology

```
Layer 1 (evidence, 6 nodes):  Budget  GroupSize  ActivityLevel  Setting  TimeOfDay  Interest
Layer 2 (latent, 3 nodes):        SocialContext      EnergyProfile        VenueType
Layer 3 (output, 1 node):                          EventRec  (8 events)
```

**Arcs (parent sets)** — each *pair* of evidence nodes drives one interpretable
latent factor:

- `VenueType     ← Interest, Setting`
- `SocialContext ← GroupSize, Budget`
- `EnergyProfile ← ActivityLevel, TimeOfDay`
- `EventRec      ← VenueType, SocialContext, EnergyProfile`

### 10.2 Variable domains

- **Evidence** — as in `EVIDENCE_LABELS` (§5.1).
- **Latent:**
  - `VenueType` (`VENUE`) = `Cultural`, `Entertainment`, `Nature`, `Culinary`.
  - `SocialContext` (`SOCIAL`) = `Intimate`, `Social`, `Mass`.
  - `EnergyProfile` (`ENERGY`) = `Low`, `Medium`, `High`.
- **Output** — `EventRec` (`EVENTS`) = the 8 `EVENT_LABELS`.

### 10.3 Soft rules → latent CPTs

Rather than hand-writing full CPT tables, the CPTs are **generated** from soft
rules grounded in the proposal's Event Catalogue. Each latent factor has a
**base** distribution (from one parent) multiplied by a **modifier** (from the
other), then normalised (`_combine(base, modifier, order)` →
`_normalised({k: base*modifier})`).

**VenueType ← Interest (base), Setting (modifier):**

`_INTEREST_VENUE` (base, rows sum to 1):

| Interest | Cultural | Entertainment | Nature | Culinary |
|---|---|---|---|---|
| Arts | 0.70 | 0.15 | 0.05 | 0.10 |
| Music | 0.15 | 0.70 | 0.05 | 0.10 |
| Food | 0.10 | 0.10 | 0.10 | 0.70 |
| Sports | 0.05 | 0.55 | 0.35 | 0.05 |

`_SETTING_VENUE` (multiplicative modifier):

| Setting | Cultural | Entertainment | Nature | Culinary |
|---|---|---|---|---|
| Indoor | 1.2 | 1.1 | 0.4 | 1.1 |
| Outdoor | 0.7 | 0.9 | 2.5 | 0.9 |
| Either | 1.0 | 1.0 | 1.0 | 1.0 |

**SocialContext ← GroupSize (base), Budget (modifier):**

`_GROUP_SOCIAL`:

| GroupSize | Intimate | Social | Mass |
|---|---|---|---|
| Solo | 0.75 | 0.20 | 0.05 |
| Small | 0.40 | 0.50 | 0.10 |
| Large | 0.05 | 0.45 | 0.50 |

`_BUDGET_SOCIAL` (modifier):

| Budget | Intimate | Social | Mass |
|---|---|---|---|
| Low | 1.1 | 1.0 | 0.9 |
| Med | 1.0 | 1.0 | 1.0 |
| High | 1.0 | 1.05 | 1.1 |

**EnergyProfile ← ActivityLevel (base), TimeOfDay (modifier):**

`_ACTIVITY_ENERGY`:

| ActivityLevel | Low | Medium | High |
|---|---|---|---|
| Relaxed | 0.70 | 0.25 | 0.05 |
| Moderate | 0.20 | 0.60 | 0.20 |
| Active | 0.05 | 0.30 | 0.65 |

`_TIME_ENERGY` (modifier):

| TimeOfDay | Low | Medium | High |
|---|---|---|---|
| Day | 1.1 | 1.0 | 0.9 |
| Evening | 1.0 | 1.1 | 1.0 |
| Night | 0.8 | 1.0 | 1.3 |

### 10.4 Event profiles → the EventRec CPT

`_EVENT_PROFILE` records, for each event, which latent values it prefers:

| Event | venue | social | energy |
|---|---|---|---|
| Museum | Cultural | Intimate, Social | Low |
| Concert | Entertainment | Social, Mass | Medium |
| Sports | Entertainment | Mass | High |
| Food | Culinary | Social, Mass | Low, Medium |
| Outdoor | Nature | Intimate | High |
| Nightlife | Entertainment | Social | High |
| Workshop | Cultural | Intimate | Medium |
| Networking | Cultural, Entertainment | Social | Medium |

**`_event_distribution(v, s, e)`** — for a latent combination, each event gets a
score `vm * sm * em + 1e-3`, where each factor is `1.0` if the latent value is in
the event's preferred set for that dimension, else a penalty (`0.20` for venue,
`0.25` for social, `0.25` for energy). The `1e-3` floor avoids hard zeros. The 8
scores are normalised into a distribution.

### 10.5 Network construction — `build_network()`

1. **Layer 1** — add each evidence node (`_make_var` builds a
   `gum.LabelizedVariable` with the labels) and fill its CPT with a **uniform
   prior** (`1/len(labels)`), so a missing/unobserved evidence node is neutral.
2. **Layer 2** — add `VenueType`, `SocialContext`, `EnergyProfile`.
3. **Layer 3** — add `EventRec`.
4. **Arcs** — as in §10.1.
5. **Latent CPTs** — for every (Interest×Setting), (GroupSize×Budget),
   (ActivityLevel×TimeOfDay) combination, fill the corresponding latent CPT column
   with the combined+normalised distribution.
6. **EventRec CPT** — for every (VenueType×SocialContext×EnergyProfile)
   combination (4×3×3 = 36 columns), fill with `_event_distribution(v, s, e)`.

`_make_var(name, labels)` and `_normalised(scores, order)` are the small helpers.

### 10.6 `BayesianRecommender` (implements `contracts.Recommender`)

- Constructor builds the network once (`self.bn = build_network()`).
- **`_inference(evidence)`** — creates a `gum.LazyPropagation(self.bn)`, cleans the
  evidence with `validate_evidence` (**the whitelist again** — third layer of
  defence), calls `setEvidence(clean)` if non-empty, `makeInference()`, and
  returns the inference engine. Partial/empty evidence is fine (unobserved nodes
  fall back to their priors).
- **`recommend(evidence) -> Result`** — posterior over `EventRec`; builds
  `[{"event", "prob"}, ...]`, sorts descending by probability. Returns all 8
  (top-3 are presented by the FSM/behaviour).
- **`explain(event, evidence) -> str`** — runs inference, reads back the **argmax**
  latent value for each of `VenueType`/`SocialContext`/`EnergyProfile`
  (`_map(node, order)`), and phrases: *"I suggest {event} because your preferences
  point to {venue} venue with a {social} vibe and a {energy} feel."* using the
  human-readable descriptors:
  - `_VENUE_DESC`: Cultural→"a cultural", Entertainment→"an entertainment",
    Nature→"an outdoor", Culinary→"a culinary".
  - `_SOCIAL_DESC`: Intimate→"intimate", Social→"social", Mass→"lively crowd".
  - `_ENERGY_DESC`: Low→"low-key", Medium→"moderate", High→"high-energy".

### 10.7 `__main__` demo

Runs four sample evidence dicts (a full one, two partial ones, and the empty dict
= prior ranking), printing the top-3 with percentages and the explanation. This
is how you sanity-check the network standalone.

### 10.8 Tests (`tests/test_recommender.py`)

Seven tests against a shared `BayesianRecommender`:

- `test_returns_all_events_as_normalised_distribution` — empty evidence yields all
  8 events, probabilities sum to 1, sorted descending.
- `test_arts_solo_relaxed_favours_museum` — Museum in top-2.
- `test_food_outdoor_favours_food` — Food is #1.
- `test_active_night_large_sporty_is_high_energy` — Sports or Nightlife in top-3.
- `test_partial_evidence_supported` — single slot still gives a valid distribution.
- `test_explain_mentions_event` — the explanation contains the event name.
- `test_invalid_values_ignored` — illegal values (`Budget="Bananas"`, `Bogus="x"`)
  don't crash and don't change legal behaviour (the whitelist at work).

---

## 11. WP4 — The behaviour / presentation layer

Three modules under `src/behaviour/`: `tts.py` (speech), `pepper_behaviour.py`
(gestures + coordination), `event_display.py` (tablet image).

### 11.1 `tts.py` — the text-to-speech worker

Speech is queued and spoken on a dedicated **worker thread** so it can run
alongside gestures. Backends in order of preference (override with `TTS_BACKEND`):

- **`_EdgeBackend`** (`edge-tts`) — Microsoft Edge neural voices (natural
  prosody). Needs internet + `edge_tts` + `playsound`. Constructor takes
  `(voice, rate="+0%", pitch="+0Hz")`; `try_create()` resolves them from
  `TTS_VOICE` (default `en-GB-RyanNeural`), `TTS_RATE` (default `-8%`), and
  `TTS_PITCH` (default `-5Hz`) — a slightly slowed, slightly lowered tuning of
  the stock Ryan voice, chosen (over cloning any specific real voice, which was
  explicitly ruled out) as the closest legitimate match to the agent's intended
  composed, measured character. `speak(text)` synthesises to a temp `.mp3` via
  `edge_tts.Communicate(text, self.voice, rate=self.rate,
  pitch=self.pitch).save(...)` inside `asyncio.run`, plays it, and always removes
  the temp file. `try_create()` returns `None` if `edge_tts` or `playsound` is
  missing.
- **`_Pyttsx3Backend`** (`pyttsx3`) — offline SAPI5/espeak fallback (more
  monotone, always available). Creates a **fresh engine per utterance** (reliable
  across repeated calls, avoiding the "run loop already started" issue).

**`_play_audio(path)`** — tries `playsound(path, True)`; on failure, if the file
is a `.wav`, falls back to `winsound.PlaySound`; else re-raises.

**`_select_backend()`** — reads `TTS_BACKEND` (`auto`/`edge`/`pyttsx3`/`none`).
`none` → no speech. `auto`/`edge` → try Edge first; if that fails and mode allows,
try pyttsx3. If the primary is a non-pyttsx3 backend, it also creates a pyttsx3
**runtime fallback** used if the neural backend errors mid-run.

**`TtsEngine`** — the worker. On init selects `(backend, fallback)`; `available`
is whether a backend exists; starts a daemon thread `_run` that pulls
`(text, done_event)` tuples from a `queue.Queue`, calls `backend.speak(text)`, and
on exception tries the fallback, always setting `done`. On shutdown it closes both
backends. **`speak(text, block=True, timeout=30.0)`** enqueues and (if blocking)
waits on the done event. `shutdown()` stops the thread (2 s join).

### 11.2 `pepper_behaviour.py` — gestures + coordination

`PepperBehaviour(pepper, speak_aloud=True)` — holds the qiBullet `pepper`, a
`TtsEngine` (if speaking aloud), and an `EventImageDisplay`. Tracks the most recent
gesture thread so gestures run concurrently with speech but are serialised among
themselves.

**Speech:**
- `say(text)` — prints `[Pepper] (says) ...` and, if TTS is available,
  `self._tts.speak(text)` (blocks until the line finishes).
- `shutdown()` — waits for any in-flight gesture, closes the display, shuts down
  TTS (call before stopping the sim).
- `clear_display()` — hides the tablet overlay.

**Gestures:**
- `gesture(name)` — looks up the handler for `wave`/`nod`/`think`/`present`/
  `open_arms`/`talk`; unknown names print a "no joint mapping — skipped" note. Waits for
  any previous gesture to finish (`wait_for_gesture`), then launches the handler on
  a **daemon thread** (returns immediately so it plays alongside speech). Handler
  exceptions are caught and logged.
- `wait_for_gesture()` — joins the current gesture thread if alive.
- **`say_with_gesture(text, name)`** — the coordination primitive: launches the
  gesture thread, speaks (blocking) concurrently, then waits for the gesture — so
  it returns only once **both** have finished. This is what makes each greeting
  beat land as a coordinated unit.

**The gesture joint sequences** (via `pepper.setAngles(joint(s), angle(s),
speed)`, all times in seconds):

- **`_wave`** — raise both arms to shoulder height
  (`LShoulderPitch/RShoulderPitch=1.4`, rolls ±0.1, elbow rolls ∓0.5), sleep 0.8;
  extend the right arm up (`RShoulderPitch=-0.5`, `RShoulderRoll=-0.5`,
  `RElbowRoll=1.0`), sleep 1.0; oscillate the right elbow 3× (0.5↔1.2), 0.4 s each;
  then `_relax_arms`.
- **`_nod`** — pitch the head down/up twice (`HeadPitch` 0.35↔-0.1, 0.4 s each),
  then centre (0.0).
- **`_think`** — bring the right hand toward the chin (`RShoulderPitch=0.6`,
  `RShoulderRoll=-0.2`, `RElbowRoll=1.5`, `RElbowYaw=1.0`) and tilt the head
  (`HeadPitch=0.2`, `HeadYaw=0.2`), sleep 1.5, recentre the head, `_relax_arms`.
- **`_present_gesture`** — open-palm presentation: right arm extended forward
  (`RShoulderPitch=0.5`, `RShoulderRoll=-0.2`, `RElbowRoll=0.4`, `RElbowYaw=0.5`),
  `openHand("RHand")` (guarded), sleep 1.5, `_relax_arms`.
- **`_open_arms`** — a two-handed gesture used for the third greeting beat:
  forearms raised so the open hands sit near the shoulders with elbows down
  (`RShoulderPitch/LShoulderPitch=1.0`, rolls ∓0.15, elbow rolls ±1.5, elbow yaws
  ±0.3), open both hands, sleep 0.6; then a few small up/down swings of both
  shoulders (0.9↔1.05, 0.3 s each), close both hands, `_relax_arms`. Total
  duration ≈3.2 s.
- **`_talk`** (requirement #6, optional) — a brief single-beat gesture used for
  each conversation question: raise both arms (0.35 s speed), open the right
  hand, sleep 0.45 s, close the right hand, `_relax_arms`. Total duration
  ≈1.25 s. This exists specifically because `_open_arms`'s ≈3.2 s swing drifted
  visibly out of sync with the dialogue's LLM-generated questions, which vary in
  length turn to turn — `say_with_gesture` blocks the gesture for exactly as long
  as the line takes to speak, so a fixed ≈3.2 s gesture either finished early
  (leaving Pepper motionless while still talking) or ran long past a short
  question. `_talk` is short enough that its natural variance in perceived timing
  stays small regardless of question length, and `main.py`'s dialogue `speak()`
  callback (§8.4) uses it in place of `open_arms` specifically for conversation
  questions; the greeting still uses `open_arms` for its third beat.
- **`_relax_arms`** — return to a neutral rest pose (`ShoulderPitch=1.4`, rolls
  ∓0.1, elbow rolls ±0.5), sleep 0.8.

**Presentation:**
- `present(result)` — prints the top events; for the top event calls
  `self._display.show_event(top_event)` (which returns the image path, printed as
  `[Pepper] (tablet) ...`); then prints the top-3 with percentages.

### 11.3 `event_display.py` — the tablet image display

qiBullet exposes Pepper's `Tablet_frame` link but no high-level tablet API, so this
helper attaches a thin textured panel to that link and swaps its texture to the
recommended event's image. If the simulator path is unavailable, it falls back to
an OpenCV window.

**AVIF decode support.** Two of the eight event images (`workshop.avif`,
`food.avif`) are AVIF files, which neither stock Pillow nor OpenCV can decode —
both silently fail to open them, so every conversion path below falls through
and nothing renders, even though `show_event()` still returns a truthy path
(the file exists on disk; only the *decode* fails). The module therefore
imports `pillow_avif` (from `pillow-avif-plugin`, §4.2/§18) at the top,
guarded in a `try/except` so its absence degrades gracefully rather than
crashing: importing it registers an AVIF decoder with `PIL.Image.open`, which
both `_make_tablet_texture` and `_convert_with_pillow` (below) rely on.

**`EVENT_IMAGE_FILES`** — maps each event to its image filename in `imgs/`:
Museum→`mueseum.jpg` (kept to match the existing filename), Concert→`concert.jpg`,
Sports→`sports.jpg`, Food→`food.avif`, Outdoor→`outdoor.jpg`,
Nightlife→`nightlife.jpg`, Workshop→`workshop.avif`, Networking→`networking.jpg`.

**`EventImageDisplay(pepper=None, image_dir=None, window_fallback=True)`:**
- `image_dir` defaults to `<repo>/imgs` (computed relative to the module).
- `path_for_event(event)` — resolves the image path if the file exists.
- **`show_event(event)`** — resolve the path; try `_show_on_pepper_tablet(path)`;
  if that fails and `window_fallback` is on, `_show_in_window(path, event)`.
  Returns the path.
- `clear()` — makes the tablet panel transparent (`changeVisualShape` with
  `rgbaColor=[1,1,1,0]`) and destroys the fallback window.
- `close()` — stops the follow thread, deletes converted temp textures, destroys
  the window.

**qiBullet display path:**
- `_show_on_pepper_tablet(path)` — imports `pybullet` lazily; converts the image
  to a texture-compatible path; ensures the panel exists; `loadTexture` +
  `changeVisualShape(..., textureUniqueId=..., rgbaColor=[1,1,1,1])`. Returns
  success/failure (falling back to a window on failure).
- `_ensure_panel(p)` — creates the panel once: a `GEOM_MESH` visual from
  `tablet_panel.obj` with `meshScale=[1.0, 0.112, 0.063]`, a zero-mass multibody,
  then looks up the `Tablet_frame` link and starts the follow thread. Raises if the
  Pepper model has no `Tablet_frame` link.
- `_start_following_tablet(p, link_index)` — a daemon thread that, at ~33 Hz
  (`sleep(0.03)`), reads the tablet link state (`getLinkState(...,
  computeForwardKinematics=True)`), computes the link's outward normal from the
  orientation quaternion, offsets the panel ~1.8 cm along that normal (so the
  image sits *in front of* the built-in black tablet rather than being hidden by
  it), and `resetBasePositionAndOrientation`s the panel to track the link.

**Texture conversion:**
- `_texture_compatible_path(path)` — tries `_make_tablet_texture`; else if the file
  is already `.jpg/.jpeg/.png`, uses it directly; else `None`.
- `_make_tablet_texture(path)` — with **Pillow**, produces a sharp 16:9 PNG
  (2048×1152): convert to RGB, centre-crop to the target aspect ratio, resize with
  LANCZOS, apply an `UnsharpMask(radius=1.2, percent=140, threshold=3)`, save PNG.
  Falls back to `_convert_with_cv2` if Pillow is unavailable or errors.

**Window fallback:**
- `_show_in_window(path, event)` — `cv2.imshow("Pepper recommendation", img)` (via
  a cv2 read, converting first if needed), titled with the event name.
- `_convert_with_cv2` / `_convert_with_pillow` — convert unusual formats (e.g.
  `.avif`) to a temp `.png`/`.rgba` that OpenCV/PyBullet can load. Pillow is tried
  first, then OpenCV.

---

## 12. WP5 — Integration: the process bridge

The robot (3.8) and dialogue (3.11) can't share an interpreter, so they run as two
processes connected by a bridge exchanging **newline-delimited JSON over stdio**.

```
robot FSM (3.8)  --collect-->  dialogue service (3.11, LangGraph + Groq)
   ^  speaks question (WP4)  <--ask--          |
   |  sends typed/ASR answer  --answer-->       |
   '------ Evidence  <--evidence--  collect_evidence() finishes
```

### 12.1 The wire protocol

**Service → robot** (only protocol JSON on stdout; all logs go to stderr):

- `{"event": "ready"}` — handshake, sent once at startup.
- `{"event": "ask", "question": "..."}` — the robot should speak it + answer.
- `{"event": "listening"}` — (speech mode) the robot prints a "speak now" cue.
- `{"event": "heard", "text": "..."}` — (speech mode) what Whisper transcribed.
- `{"event": "notify", "message": "..."}` — a status/clarification line to speak.
- `{"event": "evidence", "evidence": {...}}` — the final collected evidence.
- `{"event": "error", "message": "..."}` — a failure during collection.

**Robot → service:**

- `{"cmd": "collect"}` — start a collection run.
- `{"cmd": "quit"}` — shut down the service.
- `{"answer": "..."}` — a typed answer to the last `ask`.
- `{"spoken": true}` — (speech mode) Pepper has finished speaking; record + Whisper.

### 12.2 Robot-side client — `src/dialogue_bridge.py`

Implements the `DialogueManager` contract on the 3.8 side by spawning the 3.11
service and driving it over JSON stdio.

**`DialogueBridge(speak, dialogue_dir, answer_mode="text", get_text=None,
notify=None, on_heard=None)`:**
- `speak` voices a question (Pepper). `answer_mode="text"` → `get_text()` supplies
  the typed reply; `"speech"` → the service records the mic and transcribes
  locally.
- **`_service_cmd(dialogue_dir)`** — locates the service interpreter: prefers the
  dialogue venv's `python.exe` (Windows) / `bin/python` (POSIX) running `-m
  dialogue.bridge`; else falls back to `uv run python -m dialogue.bridge` if `uv`
  is on PATH; else raises with a helpful message ("run `uv sync` in the dialogue
  project").
- **`_spawn`** — `subprocess.Popen(cmd, cwd=dialogue_dir, stdin/stdout/stderr=PIPE,
  text=True, bufsize=1)`.
- **`_drain_stderr`** — a daemon thread that reads the service's stderr so it can't
  fill the pipe and deadlock, surfacing lines at debug level.
- **`_send`/`_recv`** — write `json.dumps(obj) + "\n"` + flush; read a line, skip
  blanks/non-JSON noise, raise if the service exited (`""`).
- **`_await_ready`** — reads the first message and asserts `event == "ready"`.
- **`collect_evidence()`** — sends `{"cmd": "collect"}`, then loops on incoming
  events: `ask` → `speak(question)`, then in speech mode send `{"spoken": true}`
  else send `{"answer": get_text()}`; `listening` → print a cue; `heard` →
  `on_heard(text)`; `notify` → `notify(message)`; `evidence` → **return** the dict;
  `error` → log + return `{}`. **Any exception** → log + return `{}` (so the BN
  still runs on whatever was gathered).
- **`close()`** — best-effort `{"cmd": "quit"}` then `proc.terminate()`.

### 12.3 Robustness of the bridge

Every failure mode (missing venv, `uv` absent, crashed service, malformed message,
mid-run exception, service error event) is caught and yields **empty evidence**,
so the recommender still runs. If the bridge can't even start, `main.py` falls back
to the console stub (§8.4). This realises proposal §6 ("some preferences missing →
run the BN with available evidence").

---

## 13. The stubs (development scaffolding)

File: `src/stubs.py`. Drop-in stand-ins that satisfy the contracts so WP1 can run
end-to-end from day one.

- **`ConsoleBehaviour`** (WP4 stub) — prints what Pepper would say/do:
  `say`/`gesture`/`say_with_gesture` print to the terminal; `clear_display` is a
  no-op; `present` prints the top-3 events with percentages.
- **`StubDialogueManager`** (WP2 stub) — `collect_evidence()` returns either a
  fixed `scripted` dict, a hard-coded `_default()` (Budget=Low, GroupSize=Small,
  ActivityLevel=Relaxed, Setting=Outdoor, TimeOfDay=Day, Interest=Food), or (when
  `interactive=True`) asks one multiple-choice question per slot on the console,
  keeping only valid answers.
- **`StubRecommender`** (WP3 stub) — a toy deterministic ranker: maps `Interest`
  to a couple of likely events (`_INTEREST_BIAS`), nudges `Outdoor` up if
  `Setting=Outdoor`, orders the rest, and assigns linearly decreasing
  pseudo-probabilities that sum to 1. `explain` stitches a "because ..." string
  from Interest/Setting/Budget. This is what `main.py` uses if the real
  `BayesianRecommender` can't be constructed.

---

## 14. End-to-end data-flow walkthrough

A full `--source hybrid --dialogue real --answer speech` encounter:

1. **Startup** — `main.py` spawns Pepper in qiBullet, builds a shared
   `FaceRecognizer()` (§6.7), `PepperBehaviour` (TTS worker + tablet display), and
   a `ThreadedPerception(HaarFaceDetector(WebcamSource, recognizer=recognizer))`
   preview. `_build_real_dialogue` spawns the dialogue service via
   `DialogueBridge`; the service emits `ready`. `BayesianRecommender` builds the
   network. Prints the `"=== JARVIS - At your service. ... (source=hybrid) ==="`
   banner. The FSM starts in **Idle**.
2. **Idle → Greeting** — the background perception thread fills its window; once ≥4
   of 5 recent frames contain a face, `face_stable()` flips true. `_idle()` logs
   whether `perception.identify()` recognized the face (informational only) and
   the FSM enters **Greeting**.
3. **Greeting** — `_identify_or_enroll()` runs first (§7.4): if the face matches
   someone in `known_faces/`, their name comes back immediately; otherwise JARVIS
   asks *"I don't think we've met yet. What's your name?"*, then asks consent to
   remember the face, and — if granted — captures ~15 live samples through the
   same perception object and retrains the recognizer on the spot. Either way a
   name (or `None`) comes out, feeding the personalised or generic greeting line.
   Three `say_with_gesture` beats follow (wave / nod / open_arms), each speaking
   (neural TTS, tuned Ryan voice) and moving concurrently and blocking until both
   finish. Then a presence check; if the user's still there → **Conversation**.
4. **Conversation** — `dialogue.collect_evidence()` drives the bridge: the service
   runs the LangGraph. Per turn: `Router` picks the next empty slot **at random**
   (so the opening question isn't always the same one run to run) →
   `QuestionFramer` asks a Groq-generated open question, nudged by a randomly
   chosen style hint toward a composed, understated tone → the bridge's `speak()`
   callback voices it with the brief, question-length-synced `"talk"` gesture → the
   robot sends `{"spoken": true}` → the service records the mic and transcribes
   with **local Whisper** (`heard`) → `AnswerParser` first checks the raw reply
   against the small abuse-word list (§9.5.2); an abusive reply short-circuits to
   a polite "let's keep this friendly" notification (spoken with a `"nod"` gesture)
   instead of being parsed. Otherwise it runs the Groq parser (with the
   weak-value guard) → `ConflictResolver` handles contradictions (re-open +
   either/or) → `Evaluator` commits / reframes / defaults. One rich reply may fill
   several slots. When all six are filled (or defaulted), `Finish` returns the
   evidence; the service emits `{"event": "evidence", ...}`; the bridge returns
   it. The FSM applies `validate_evidence` again.
5. **Reasoning** — `say_with_gesture("Let me think about that.", "think")`; the
   `BayesianRecommender` runs LazyPropagation over the (partial) evidence and
   returns the 8 events ranked.
6. **Recommendation** — `say_with_gesture("Based on what you told me...",
   "present")`; `behaviour.present(result)` shows the top event's image on the
   tablet panel (which tracks the `Tablet_frame` link, decoding AVIF sources via
   `pillow-avif-plugin` where needed) and prints the top-3; `recommender.explain(
   top, evidence)` reads back the argmax latent factors into a "because..."
   sentence, which JARVIS speaks. Then `_wait_for_ack()` waits up to `ack_timeout`
   or until the user leaves.
7. **Farewell** — `say_with_gesture("Enjoy your evening. JARVIS signing off,
   goodbye.", "wave")`; `_reset_interaction()` clears the tablet, evidence,
   result, and the perception window. Back to **Idle** for the next person.

At every step, if `_still_present()` reports the user gone, the FSM short-circuits
back to Idle.

---

## 15. Configuration reference

### 15.1 CLI — `src/main.py`

See §8.1 (source/camera-index/preview/headless/interactive/dialogue/answer/
max-cycles/verbose).

### 15.2 CLI — `dialogue.cli`

`--llm {groq,scripted}`, `--input {text,speech,scripted}`, `--no-frame`,
`--verbose`.

### 15.3 CLI — `check_perception.py`

`--source {webcam,pepper}`, `--camera-index N`, `--headless`, `--no-window`.

### 15.4 CLI — `enroll_face.py`

`--name NAME` (required), `--count N` (default `20`), `--camera-index N`
(default `0`), `--interval SECONDS` (default `0.3`). See §6.8.

### 15.5 Environment variables

**Dialogue / LLM:**

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` / `GROK_KEY` | — | Groq API key (either name accepted) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `LANGSMITH_API_KEY` | — | LangSmith tracing key (exact name required) |
| `LANGSMITH_TRACING` | — | `"true"` to enable tracing |
| `LANGSMITH_PROJECT` | — | LangSmith project name |

**Text-to-speech (`tts.py`):**

| Variable | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `auto` | `edge` (neural) · `pyttsx3` (offline) · `none` |
| `TTS_VOICE` | `en-GB-RyanNeural` | any Edge neural voice |
| `TTS_RATE` | `-8%` | Edge neural speaking-rate offset (e.g. `-8%`, `+10%`) |
| `TTS_PITCH` | `-5Hz` | Edge neural pitch offset (e.g. `-5Hz`, `+10Hz`) |

**Local Whisper ASR (`asr.py`):**

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` (GPU) / `small` (CPU) | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `auto` | `cuda` / `cpu` |
| `WHISPER_COMPUTE` | `int8_float16` (GPU) / `int8` (CPU) | `float16` / `int8_float16` / `int8` |
| `WHISPER_MIC_INDEX` | — | input device index if the default mic is wrong |
| `WHISPER_ENERGY` | — | fixed energy threshold (else calibrated from ambient noise) |

For a fully offline run: `TTS_BACKEND=pyttsx3` (neural TTS uses the network; local
Whisper runs on-device; only the LLM text calls go to Groq).

---

## 16. Testing

- **`tests/test_recommender.py`** (robot side) — the seven WP3 tests in §10.8. Run
  from `Project/` with `uv run pytest` (`pythonpath=["src"]`).
- **`dialogue/tests/test_graph_offline.py`** (dialogue side) — three offline graph
  tests using `ScriptedParser` + `ScriptedInput` (no network, no mic):
  - `test_collects_all_slots_from_clear_answers` — six clear answers fill all six
    slots exactly.
  - `test_multi_slot_fill_from_one_answer` — "cheap, outdoors, with a few friends"
    fills Budget+Setting+GroupSize from one reply; the rest asked normally; all six
    end up filled.
  - `test_default_applied_after_failed_clarifications` — gibberish for Budget three
    times → the `Med` default is applied after `max_clarify=2`; later answers parse
    fine; every value is legal.
  Run from `Project/dialogue/` with `uv run pytest -q`.

Both suites are designed to run with **no robot, no simulator, no live LLM, no
microphone** — the scripted parser/input and the stub recommender make the whole
pipeline exercisable deterministically.

---

## 17. Robustness & fallback matrix

| Situation | Handling | Where |
|---|---|---|
| Face flickers on/off | 5-frame smoothing window; strict `face_stable` (4/5) to start, loose `is_present` (2/5) to sustain | `HaarFaceDetector` |
| Preview window freezes during dialogue | perception runs on a background thread | `ThreadedPerception` |
| Windows webcam won't stream (MSMF bug) | prefer DirectShow, verify with a real `read()` | `WebcamSource._open` |
| User gives a vague/agreeable answer | LLM prompt omits it; weak-value guard drops catch-alls without a no-preference cue | `llm.py`, `graph.AnswerParser` |
| Answer contradicts an earlier one | ConflictResolver re-opens the slot and asks an either/or | `graph.ConflictResolver` |
| Answer never parses | reframe up to `max_attempts`, then apply the slot default + notify | `graph.Evaluator` |
| Some preferences missing | BN runs on partial evidence; unobserved nodes use priors | `BayesianRecommender._inference` |
| LLM emits an illegal value | whitelisted out at three layers (parser, robot `validate_evidence`, BN inference) | `schema.whitelist`, `contracts.validate_evidence` |
| No Groq key / LLM call fails | fall back to `ScriptedParser` (keyword) / return `{}` | `bridge._build_llm`, `GroqEvidenceParser.parse` |
| Question framing fails | fall back to the slot's canned question | `graph.QuestionFramer` |
| Dialogue service can't start | fall back to the console stub | `main._build_real_dialogue` |
| Bridge fails mid-run | catch → empty evidence, BN still runs | `DialogueBridge.collect_evidence` |
| Neural TTS unavailable/offline | fall back to pyttsx3 (per line) | `tts.TtsEngine._run` |
| GPU Whisper fails at runtime | reload on CPU once and stay there | `asr.WhisperTranscriber.transcribe_wav` |
| Mic capture/transcription fails | return `""` (treated as an unusable answer → reframe) | `asr.listen_and_transcribe` |
| Recommender unavailable (pyAgrum/etc.) | fall back to `StubRecommender` | `main.main` |
| Weak/empty recommendation | present top-3, or apologise gracefully if empty | `state_machine._recommendation` |
| Tablet surface unavailable | fall back to an OpenCV preview window | `EventImageDisplay` |
| AVIF image format (`workshop.avif`, `food.avif`) | decode via `pillow-avif-plugin` registering an AVIF opener with PIL (stock Pillow/OpenCV can't decode AVIF at all) | `EventImageDisplay` (module-level import), `_make_tablet_texture` |
| Unrecognized visitor face | ask for a name and explicit consent before enrolling; declining is honoured (name used for that conversation only, nothing captured) | `state_machine._identify_or_enroll` |
| No camera source wired (e.g. `--source scripted`) | face verification/enrollment is simply skipped (`recognizer` is `None`) | `state_machine._identify_or_enroll` |
| Abusive language in a reply | declined and the person is asked to rephrase, rather than being parsed as evidence | `dialogue.graph.AnswerParser` / `_is_abusive` |
| User walks away at any point | `_still_present()` short-circuits back to Idle | `state_machine` |
| Ctrl-C | caught for a clean shutdown | `InteractionFSM.run` |

---

## 18. Dependency reference

**Robot side (`Project/pyproject.toml`, Python 3.8):** `opencv-contrib-python`
(the *contrib* build, for `cv2.face` LBPH face recognition), `numpy`, `qibullet`,
`pybullet` (WP1); `pyagrum` (WP3); `pyttsx3`, `edge-tts`, `playsound`,
`threadpool`, `pillow-avif-plugin` (AVIF decode for two event images) (WP4);
`pytest` (dev). Optional at runtime: `Pillow` (used by the tablet texture
converter if present; also a transitive dependency `pillow-avif-plugin` builds
on).

**Dialogue side (`Project/dialogue/pyproject.toml`, Python 3.11):** `langgraph`,
`openai`, `python-dotenv` (core); `SpeechRecognition`, `pyaudio`, `faster-whisper`,
`nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (extra `speech`); `langgraph-cli[inmem]`,
`pytest` (dev).

**External services:** Groq (LLM inference over the OpenAI-compatible API);
LangSmith (optional tracing). Everything else — face detection, Bayesian
reasoning, speech recognition, and (with pyttsx3) speech synthesis — runs locally.

---

## 19. Glossary

- **Evidence** — the dict of up to six discrete preference values; the single data
  currency between packages. Partial evidence is allowed.
- **Slot** — one preference field (Budget, GroupSize, …) with a legal value set,
  a canned question, a clarify prompt, and a default.
- **Whitelist / validate** — the guard that keeps only legal slot/value pairs,
  preventing the LLM from injecting anything the Bayesian network can't interpret.
- **Latent factor** — an interpretable intermediate variable (VenueType,
  SocialContext, EnergyProfile) that turns raw evidence into an explainable basis
  for the recommendation.
- **CPT** — Conditional Probability Table; here generated from soft, human-readable
  rules rather than hand-tuned numbers.
- **LazyPropagation** — pyAgrum's exact inference engine used to compute the
  posterior over events (and the latent argmaxes for the explanation).
- **Framer** — the LLM component that generates each next question conversationally.
- **LBPH** — Local Binary Patterns Histograms; the on-device face-recognition
  algorithm (`cv2.face.LBPHFaceRecognizer_create()`) `FaceRecognizer` uses to
  verify a detected face against enrolled photos. Its confidence score is a
  distance (lower = better match), the inverse of a typical classifier score.
- **Consent-based enrollment** — the live flow where JARVIS asks an unrecognized
  visitor's name and explicit permission before capturing and saving face
  samples; declining is honoured and nothing is captured.
- **Abuse filter** — the small flagged-word substring check (`_ABUSE_WORDS` /
  `_is_abusive`) that stops an abusive reply from ever reaching the LLM parser or
  evidence, asking the person to rephrase instead.
- **Style hint** — one of a small set of tone directives randomly chosen each
  `GroqQuestionFramer.frame()` call, so repeated questions don't converge on the
  same phrasing/angle.
- **Bridge** — the subprocess + JSON-over-stdio link between the 3.8 robot and the
  3.11 dialogue service.
- **WP1–WP5** — the five work packages: perception/FSM, dialogue, recommender,
  behaviour, integration.

---

*This document reflects the source as implemented in `Project/`. When code changes,
update the relevant section — especially the CPT tables (§10), the gesture joint
angles (§11.2), the wire protocol (§12.1), and the configuration reference (§15).*
