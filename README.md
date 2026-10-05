<div align="center">

# Social Event Recommendation Agent for the Pepper Robot

*Face perception, an LLM-driven conversation and an explainable Bayesian network,<br/>
running on Pepper in the qiBullet simulator*

![Python 3.8](https://img.shields.io/badge/robot-Python%203.8-3776AB?logo=python&logoColor=white)
![Python 3.11](https://img.shields.io/badge/dialogue-Python%203.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-dialogue%20graph-1C3C3C?logo=langchain&logoColor=white)
![pyAgrum](https://img.shields.io/badge/pyAgrum-Bayesian%20network-6A5ACD)
![qiBullet](https://img.shields.io/badge/qiBullet-Pepper%20simulation-E4572E)
![OpenCV](https://img.shields.io/badge/OpenCV-face%20detection-5C3EE8?logo=opencv&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)

[Overview](#overview) ·
[Demo](#demo) ·
[Architecture](#architecture) ·
[LangGraph dialogue](#the-conversation-a-langgraph-agent-graph) ·
[Quick start](#quick-start) ·
[Documentation](#documentation)

**Read the docs →**
[Robot runtime](src/README.md) ·
[Perception](src/perception/README.md) ·
[Dialogue](dialogue/README.md) ·
[Recommender](src/recommender/README.md) ·
[Behaviour](src/behaviour/README.md)

</div>

---

## Overview

This agent runs on the **Pepper** humanoid robot (in the **qiBullet** simulator) and
recommends a social event to the person standing in front of it.

When someone approaches, Pepper recognises their face, greets them with
coordinated speech and gestures, and asks a few open questions about what kind
of outing they are in the mood for. A **Bayesian network** then ranks eight
event types. Pepper says its recommendation out loud, shows a matching picture
on its tablet, and explains *why* it chose it.

The design rule throughout: **the user speaks naturally, but the decision stays
inspectable.**

| | Capability | How | Details |
|---|---|---|---|
| **Sees** | Notices a visitor and recognises returning faces | OpenCV Haar cascade with temporal smoothing, LBPH face verification | [Perception →](src/perception/README.md) |
| **Listens** | Understands free-form typed or spoken replies | Local Whisper ASR, Groq-hosted LLM parser | [Dialogue →](dialogue/README.md#speech-input) |
| **Converses** | Asks open, varied questions and handles vague or contradictory answers | LangGraph multi-agent slot-filling graph | [Dialogue →](dialogue/README.md) |
| **Reasons** | Produces an explainable, ranked recommendation | 3-layer pyAgrum Bayesian network | [Recommender →](src/recommender/README.md) |
| **Acts** | Speaks, gestures and shows the result | Neural TTS, Pepper joint control, tablet texture | [Behaviour →](src/behaviour/README.md) |
| **Stays safe** | Keeps LLM output legal, asks consent before storing faces, filters abuse | Three-layer whitelist, consent-gated enrollment, abuse filter | [Safety →](dialogue/README.md#from-free-text-to-safe-evidence) |

---

## Documentation

Every part of the system has its own README with diagrams. Click a card to open it.

<table>
  <tr>
    <td width="33%" valign="top">
      <h3><a href="src/README.md">Robot runtime →</a></h3>
      <p>The state machine, run modes, the two-process JSON bridge and every fallback</p>
      <a href="src/README.md">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/fsm-dark.svg">
          <img src="docs/diagrams/fsm.svg" alt="Robot runtime" width="100%">
        </picture>
      </a>
    </td>
    <td width="33%" valign="top">
      <h3><a href="src/perception/README.md">Perception →</a></h3>
      <p>Face detection with smoothing, face verification and consent-based enrollment</p>
      <a href="src/perception/README.md">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/perception-pipeline-dark.svg">
          <img src="docs/diagrams/perception-pipeline.svg" alt="Perception" width="100%">
        </picture>
      </a>
    </td>
    <td width="33%" valign="top">
      <h3><a href="dialogue/README.md">Dialogue →</a></h3>
      <p>The LangGraph agent graph, LLM parsing, safety guards and speech input</p>
      <a href="dialogue/README.md">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/dialogue-graph-dark.svg">
          <img src="docs/diagrams/dialogue-graph.svg" alt="Dialogue" width="100%">
        </picture>
      </a>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <h3><a href="src/recommender/README.md">Recommender →</a></h3>
      <p>Network topology, rule-generated probability tables, inference and explanations</p>
      <a href="src/recommender/README.md">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/bayesian-network-dark.svg">
          <img src="docs/diagrams/bayesian-network.svg" alt="Recommender" width="100%">
        </picture>
      </a>
    </td>
    <td width="33%" valign="top">
      <h3><a href="src/behaviour/README.md">Behaviour →</a></h3>
      <p>Speech and gesture timing, the gesture library, voices and the tablet display</p>
      <a href="src/behaviour/README.md">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/speech-gesture-dark.svg">
          <img src="docs/diagrams/speech-gesture.svg" alt="Behaviour" width="100%">
        </picture>
      </a>
    </td>
    <td width="33%" valign="top">
      <h3>More references</h3>
      <p><a href="TECHNICAL_DOCUMENTATION.md">Technical documentation</a>: every module and parameter</p>
      <p><a href="dialogue/RUNNING.md">Running and tracing the dialogue</a> in LangSmith and LangGraph Studio</p>
      <p><a href="PLAN.md">Work plan</a> · <a href="Proposal/proposal.pdf">Project proposal</a></p>
      <p><a href="docs/diagrams/build.py">Figure generator</a>: rebuilds every diagram</p>
    </td>
  </tr>
</table>

---


## Demo

| Recording | Content |
|---|---|
| [`demo/Final_Demo_SERA.mp4`](demo/Final_Demo_SERA.mp4) | A full end-to-end encounter with Pepper in qiBullet |
| [`demo/Final_SERA_langgraph.mp4`](demo/Final_SERA_langgraph.mp4) | The dialogue graph running, as seen in LangGraph |

<details>
<summary><b>Sample conversation transcript</b></summary>

```text
(a person steps into view; their face is checked against known visitors)
Pepper:  Greetings, Sunesh. JARVIS online, at your service.       (waves)
Pepper:  Systems nominal. It's good to see you.                   (nods)
Pepper:  Scanning the local grid for a social event to suit you.  (opens both arms)

Pepper:  Will it just be you this evening, or is anyone joining you?
You:     just me, keeping it simple
Pepper:  What kind of pace suits you tonight: relaxed, or something livelier?
You:     pretty relaxed
   ... a few more open questions; their order and wording change every run ...

Pepper:  Let me think about that.                                 (thinking pose)
Pepper:  Based on what you told me, I recommend the following.    (presents)
            1. Museum        41%
            2. Workshop      19%
            3. Food          11%
Pepper:  I suggest Museum because your preferences point to a cultural venue
         with an intimate vibe and a low-key feel.
         (the Museum image appears on Pepper's tablet)

Pepper:  Enjoy your evening. JARVIS signing off, goodbye.         (waves)
```

The robot's persona introduces itself as *JARVIS*. If it doesn't recognise the
visitor, it first asks their name and whether it may remember their face. If a
reply is abusive, it politely asks the person to rephrase instead of parsing it.

</details>

---

## Architecture

The agent is a single processing route, from perception to action. It is split
across **two Python runtimes**: qiBullet needs Python 3.8, while LangGraph needs
Python 3.9 or newer. The two processes exchange newline-delimited JSON over
stdio.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/architecture-dark.svg">
    <img src="docs/diagrams/architecture.svg" alt="System architecture" width="100%">
  </picture>
</p>

> **Go deeper →** [Perception](src/perception/README.md) · [Dialogue](dialogue/README.md) ·
> [Recommender](src/recommender/README.md) · [Behaviour](src/behaviour/README.md) ·
> [Robot runtime and the two-process bridge](src/README.md#the-two-runtime-bridge)

### Interaction lifecycle

Each encounter follows a six-state machine.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/lifecycle-dark.svg">
    <img src="docs/diagrams/lifecycle.svg" alt="Interaction lifecycle" width="100%">
  </picture>
</p>

> **Go deeper →** every transition, the greeting steps and what happens when a step fails:
> [the state machine in detail](src/README.md#the-interaction-state-machine)

### The conversation: a LangGraph agent graph

During the Conversation state, a LangGraph `StateGraph` of five cooperating
nodes runs the dialogue. It picks the next unknown preference, has the LLM ask
an open question, parses the reply safely, resolves contradictions, and decides
whether to move on, ask again, or fall back to a default.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/dialogue-graph-dark.svg">
    <img src="docs/diagrams/dialogue-graph.svg" alt="LangGraph dialogue manager" width="100%">
  </picture>
</p>

| See it | How |
|---|---|
| **Run it live in LangGraph Studio** | `cd dialogue && uv run langgraph dev` opens the graph in the browser; each question pauses for your answer ([guide](dialogue/RUNNING.md#4-visualize--step-through-in-langgraph-studio)) |
| **Watch the recording** | [`demo/Final_SERA_langgraph.mp4`](demo/Final_SERA_langgraph.mp4): the graph running in LangGraph |
| **Trace every run** | With `LANGSMITH_*` set, each node's input, output and LLM call shows up in LangSmith ([guide](dialogue/RUNNING.md#3-view-runs-in-langsmith-smithlangchaincom)) |
| **LangGraph's own rendering** | [`docs/dialogue_graph.png`](docs/dialogue_graph.png), exported from the compiled graph |

> **Go deeper →** every node, the safety guards and speech input:
> [Dialogue README](dialogue/README.md)

### What the agent learns and recommends

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/bayesian-network-dark.svg">
    <img src="docs/diagrams/bayesian-network.svg" alt="Bayesian recommendation network" width="100%">
  </picture>
</p>

The conversation doesn't need to fill all six. The network reasons over
whatever evidence it has, and treats the rest as unknown.

> **Go deeper →** how the probability tables are generated and how the spoken
> explanation is produced: [Recommender README](src/recommender/README.md)

<table>
  <tr>
    <td align="center"><img src="docs/img/events/museum.jpg" width="170" alt="Museum"><br/><sub><b>Museum</b></sub></td>
    <td align="center"><img src="docs/img/events/concert.jpg" width="170" alt="Concert"><br/><sub><b>Concert</b></sub></td>
    <td align="center"><img src="docs/img/events/sports.jpg" width="170" alt="Sports"><br/><sub><b>Sports</b></sub></td>
    <td align="center"><img src="docs/img/events/food.jpg" width="170" alt="Food"><br/><sub><b>Food</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/img/events/outdoor.jpg" width="170" alt="Outdoor"><br/><sub><b>Outdoor</b></sub></td>
    <td align="center"><img src="docs/img/events/nightlife.jpg" width="170" alt="Nightlife"><br/><sub><b>Nightlife</b></sub></td>
    <td align="center"><img src="docs/img/events/workshop.jpg" width="170" alt="Workshop"><br/><sub><b>Workshop</b></sub></td>
    <td align="center"><img src="docs/img/events/networking.jpg" width="170" alt="Networking"><br/><sub><b>Networking</b></sub></td>
  </tr>
</table>

<sub>The image Pepper shows on its tablet for each event.</sub>

---

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs the right Python version for each half.
- A free [Groq API key](https://console.groq.com) for the real language-model dialogue. Without one, the dialogue falls back to an offline keyword parser.
- Optional: a webcam, a microphone, and a CUDA GPU for faster Whisper.

### 1. Install

```bash
git clone https://github.com/SuneshSundarasami/Social-Event-Recommendation-Agent-on-Pepper-using-LangGraph-and-pyAgrum-BayesNet.git
cd Social-Event-Recommendation-Agent-on-Pepper-using-LangGraph-and-pyAgrum-BayesNet

uv sync                                     # robot side (Python 3.8)
cd dialogue && uv sync --extra speech       # dialogue side (Python 3.11); omit --extra speech for typed input only
cd ..
```

### 2. Configure

Create `src/.env` (it is git-ignored):

```ini
GROQ_API_KEY="gsk_..."

# Optional: trace every dialogue run at smith.langchain.com
LANGSMITH_API_KEY="lsv2_..."
LANGSMITH_TRACING="true"
LANGSMITH_PROJECT="dialogue"
```

### 3. Run

Run from the repository root. Each step up adds more of the real system:

```bash
# 1. Headless smoke test: no camera, no simulator, one scripted interaction
uv run python src/main.py --source scripted

# 2. Real LLM conversation in the terminal (Pepper's lines printed, you type)
uv run python src/main.py --source scripted --dialogue real

# 3. Pepper in the qiBullet GUI, using its own simulated head camera
uv run python src/main.py --source pepper --dialogue real

# 4. Full demo: Pepper in the GUI, webcam face detection, spoken answers
uv run python src/main.py --source hybrid --dialogue real --answer speech
```

| Flag | Choices | Meaning |
|---|---|---|
| `--source` | `scripted` · `webcam` · `pepper` · `hybrid` | Where faces come from and whether Pepper is spawned |
| `--dialogue` | `stub` · `real` | Console stand-in, or the LangGraph service through the bridge |
| `--answer` | `text` · `speech` | Typed replies, or spoken replies via local Whisper |
| `--preview` | flag | Show the webcam window with detection boxes |
| `--interactive` | flag | With the stub dialogue, ask the questions on the console |
| `--max-cycles` | int | Stop after N interactions |

See [`src/README.md`](src/README.md#run-modes) for what each source wires up,
and [`src/behaviour/README.md`](src/behaviour/README.md#configuration) and
[`dialogue/README.md`](dialogue/README.md#configuration) for the voice and
speech-recognition environment variables.

### 4. Test

```bash
uv run pytest                     # Bayesian recommender tests (robot side)
cd dialogue && uv run pytest -q   # offline dialogue-graph tests
```

Both suites run without a robot, simulator, network or microphone.

---

## Repository layout

```text
.
├── src/                    Robot runtime (Python 3.8)          → src/README.md
│   ├── main.py               entry point and component wiring
│   ├── state_machine.py      six-state interaction FSM
│   ├── contracts.py          shared types, interfaces, whitelist
│   ├── dialogue_bridge.py    client for the dialogue service
│   ├── enroll_face.py        pre-enroll a face offline
│   ├── perception/           face detection and verification   → src/perception/README.md
│   ├── recommender/          pyAgrum Bayesian network          → src/recommender/README.md
│   └── behaviour/            speech, gestures, tablet          → src/behaviour/README.md
├── dialogue/               Dialogue service (Python 3.11)      → dialogue/README.md
│   ├── dialogue/             LangGraph graph, LLM, ASR, bridge server
│   └── tests/                offline graph tests
├── tests/                  Recommender tests
├── imgs/                   Event images shown on the tablet
├── demo/                   Demo recordings
├── docs/                   Rendered figures
├── Presentation/           Project slide deck
└── Proposal/               Project proposal (LaTeX)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Simulation and control | [qiBullet](https://github.com/softbankrobotics-research/qibullet) / PyBullet |
| Vision | OpenCV (`opencv-contrib-python`): Haar cascade detection, LBPH recognition |
| Dialogue orchestration | [LangGraph](https://github.com/langchain-ai/langgraph), with LangSmith tracing and LangGraph Studio |
| Language model | [Groq](https://groq.com) through its OpenAI-compatible API (`llama-3.1-8b-instant` by default) |
| Speech recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), on-device, GPU or CPU |
| Speech synthesis | `edge-tts` neural voice, with offline `pyttsx3` fallback |
| Reasoning | [pyAgrum](https://agrum.gitlab.io/), exact inference with LazyPropagation |
| Imaging | Pillow + `pillow-avif-plugin` |
| Environments | [uv](https://docs.astral.sh/uv/), two isolated projects |

---

<div align="center">
<sub>M.Sc. Autonomous Systems · Hochschule Bonn-Rhein-Sieg · HCICR course project by Sunesh Praveen Raja Sundarasami</sub>
</div>
