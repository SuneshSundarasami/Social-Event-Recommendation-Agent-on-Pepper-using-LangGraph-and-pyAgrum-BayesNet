# Social Event Recommendation Agent — Project Plan

> Working plan and reference for the HCICR project (Pepper + qiBullet).
> Source of truth for scope, work packages, interfaces, and design decisions.
> Based on `Project/Proposal/proposal.pdf`.

---

## 1. Overview

A socially interactive recommendation agent for the **Pepper** robot in the
**qiBullet** simulation. It detects a nearby user, runs a short conversation to
elicit preferences for social activities, and recommends suitable events.

Three complementary components:

1. **Finite-state interaction manager** — controls visible robot behaviour, keeps
   the dialogue predictable.
2. **LLM interface layer** — interprets natural answers, converts them to discrete
   evidence values. It *never* changes the model structure or CPTs.
3. **Bayesian network** — transparent probabilistic reasoning over event categories.

Design principle: the user speaks naturally, but the final decision stays
**inspectable** (explicit evidence + interpretable latent factors).

### Interaction flow (six-state cycle)

| State | Robot action | Key transition |
|---|---|---|
| Idle | Passive camera monitoring | Face stable for several frames |
| Greeting | Wave, speech greeting, optional name | User remains present |
| Conversation | Preference questions w/ clarification | Required evidence complete |
| Reasoning | LLM parsing + Bayesian inference | Posterior computed |
| Recommendation | Top events, explanation, open-palm gesture | User acknowledges / timeout |
| Farewell | Goodbye speech and wave | Return to Idle |

---

## 2. Key decisions & constraints

- **Bayesian library: pyAgrum** (not pgmpy as the proposal text says). Already
  installed and working in `Homework/HW_03/bayesian_network.py`. Decision: keep
  pyAgrum unless a hard blocker appears.
- **Python split (important):**
  - Robot side (perception, behaviour, qiBullet/NAOqi) is pinned to **Python 3.8**.
  - LangGraph (dialogue manager) requires **Python ≥3.9**. → Dialogue manager runs
    as its **own process/service** (3.10+) behind a thin boundary. WP1 only ever
    calls `collect_evidence() -> Evidence`.
- **Reusable assets already in the repo:**
  - `Homework/HW_02/behaviors/` — speech, gesture, gaze, head, posture +
    `behavior_realizer` on Pepper/qiBullet → basis for WP4.
  - `Homework/HW_03/bayesian_network.py` — working pyAgrum BN → template for WP3.
  - `Homework/HW_03/rasa/` — earlier Rasa dialogue experiment (reference only;
    superseded by the LangGraph design for WP2).

---

## 3. Shared contracts (freeze these first)

Everything passes through these two types. Freezing them lets each work package be
built and tested in isolation, then snapped together.

```python
# Evidence dict — the single currency between packages.
# Any key may be MISSING (partial evidence is allowed).
Evidence = {
  "Budget":        "Low|Med|High",
  "GroupSize":     "Solo|Small|Large",
  "ActivityLevel": "Relaxed|Moderate|Active",
  "Setting":       "Indoor|Outdoor|Either",
  "TimeOfDay":     "Day|Evening|Night",
  "Interest":      "Arts|Music|Food|Sports",
}

# Recommendation result — sorted descending, top-3 presented.
Result = [{"event": "Museum", "prob": 0.31}, ...]
```

Legal event labels (8): `Museum`, `Concert`, `Sports`, `Food`, `Outdoor`,
`Nightlife`, `Workshop`, `Networking`.

Suggested repo layout (under `Project/`):

```
Project/src/
  perception/      # WP1
  dialogue/        # WP2 (separate 3.10+ process)
  recommender/     # WP3
  behaviour/       # WP4
  contracts.py     # shared Evidence/Result types + label whitelists
  main.py          # WP1 + WP5 orchestrator
```

---

## 4. Work packages

### WP1 — Perception & State Machine (the spine)
- qiBullet/Pepper setup; webcam face detection (OpenCV Haar or MediaPipe);
  5-frame smoothing window to avoid flicker.
- Six-state FSM (table in §1) + "demo interruption → Idle" reset.
- Owns the orchestration loop; calls WP2/WP3/WP4 via the contracts.
- **Produces:** FSM with hooks `on_greeting()`, `collect_evidence() -> Evidence`,
  `recommend(Evidence) -> Result`, `present(Result)`. Build with stubs first.

### WP2 — Dialogue Manager + LLM Evidence Parser  (LangGraph — see §5)
- Decides next question, tracks filled slots, clarifies ambiguous answers, applies
  defaults for missing ones.
- LLM maps free text → discrete evidence values; validation whitelists output so
  the LLM can only emit legal labels.
- **Produces:** `collect_evidence() -> Evidence`. Testable standalone with scripted
  answers (no robot, no live LLM needed).

### WP3 — Bayesian Recommendation Engine
- 3-layer pyAgrum network: 6 evidence → 3 latent (VenueType, SocialContext,
  EnergyProfile) → EventRec (8 events).
- Author CPTs from the Event Catalogue (§6); variable elimination / lazy
  propagation; partial-evidence support; ranked output + short per-event
  explanation string.
- **Produces:** `recommend(Evidence) -> Result`, `explain(event, Evidence) -> str`.
  Fully standalone; can start immediately from HW_03 template.

### WP4 — Behaviour / Presentation Layer
- Wrap `HW_02/behaviors/` into a clean API: greeting wave, attentive nod, thinking
  pose, open-palm presentation, farewell wave. Speech via ALTextToSpeech, gestures
  via ALAnimationPlayer, running concurrently with speech.
- **Produces:** `say(text)`, `gesture(name)`, `present(Result)`. Low risk (reuse).

### WP5 — Integration, Robustness & Demo ("combine later")
- Wire WP1–WP4; implement robustness table (§7); end-to-end demo tuning; the 5
  evaluation checks (§8).

### Suggested split & order
- **WP3 and WP4 are most independent — start in parallel** (reuse HW_03 / HW_02).
- WP2 develops against stub I/O.
- WP1 builds the skeleton early with stubs → WP5 integration is just swapping stubs
  for real modules.

---

## 5. WP2 design — Dialogue Manager as a LangGraph

The "ask → parse → check completeness → clarify or finish" loop maps cleanly onto a
LangGraph `StateGraph`. Runs as its own 3.10+ process behind `collect_evidence()`.

### State schema (single mutated object)

```python
class DialogueState(TypedDict):
    evidence: dict             # the 6 slots, partially filled (the contract type)
    pending_slot: str | None   # slot currently being resolved
    last_user_text: str
    clarify_count: dict        # per-slot retries, to cap clarifications
    turn_log: list             # transcript — explainability + eval evidence
    done: bool
```

### Graph shape

```
        ┌─────────────┐
        │ select_slot │◄────────────────┐
        └──────┬──────┘                 │
               │ (next missing slot)    │
        ┌──────▼──────┐                 │
        │ ask_question│                 │
        └──────┬──────┘                 │
               │ (user answer in)       │
        ┌──────▼──────┐                 │
        │ parse_llm   │  LLM → discrete │
        └──────┬──────┘     evidence    │
        ┌──────▼──────┐                 │
        │  validate   │                 │
        └──────┬──────┘                 │
        confident? │ ambiguous?         │
          ┌────────┴────────┐          │
          ▼                 ▼           │
    ┌──────────┐      ┌───────────┐    │
    │ commit   │      │ clarify   │────┘
    │ slot     │      │ (default  │
    └────┬─────┘      │  after N) │
         │            └───────────┘
   complete? ──no──────────────────────┘
         │ yes
         ▼
     ┌────────┐
     │ finish │  → returns Evidence
     └────────┘
```

### Nodes
- `select_slot` — picks next unfilled slot (ordering = question priority).
- `ask_question` — emits question text; pauses for input via `interrupt()`.
- `parse_llm` — LLM-as-interface: free text → candidate value(s) + confidence /
  `"unsure"`. May fill **multiple slots at once** ("cheap outside with friends" →
  Budget=Low, Setting=Outdoor, GroupSize=Small).
- `validate` — whitelist check; non-legal label ⇒ treat as ambiguous. Guard so the
  LLM cannot corrupt BN evidence.
- `clarify` — re-ask once; after `clarify_count[slot] >= N`, fall back to default
  and move on (matches robustness table).
- `finish` — returns the `Evidence` dict.

### LangGraph features to use
- **`interrupt()` + checkpointer** — don't block on input inside a node. Pause at
  `ask_question`, return the question; WP1 speaks it + captures answer (typed/ASR),
  resumes graph with the answer. Keeps robot I/O outside the graph; makes dialogue
  resumable (good for interruption reset).
- **Checkpointer (`MemorySaver`)** — free transcript/state snapshot; doubles as
  the "Parsing" evaluation-check evidence.

### Why over a plain `while` loop
- Completeness logic, clarification caps, multi-slot fills become explicit,
  unit-testable edges/nodes.
- Swap the LLM node for a **scripted stub** → run the whole graph in tests with no
  robot and no API calls.
- The graph diagram is the documentation.

### Open decision
Generic data-driven slot loop (compact; add/reorder questions = edit a config
table) **vs** one node per question (explicit; custom clarification prompts).
Leaning: **generic loop + per-slot config table.**

---

## 6. Bayesian model — Event Catalogue / CPT basis

Six observed evidence nodes → three latent factors → EventRec. The EventRec CPT is
based on **Layer 2 (latent)**, not directly on Layer 1. Evidence updates the latent
factors; EventRec CPT maps latent combinations to event probabilities. This keeps
the model explainable and recovers gracefully from incomplete/inconsistent answers.

| Event | VenueType | SocialContext | EnergyProfile |
|---|---|---|---|
| Museum / Gallery | Cultural | Intimate or Social | Low |
| Concert / Live Music | Entertainment | Social or Mass | Medium |
| Sporting Event | Entertainment | Mass | High |
| Food Festival | Culinary | Social or Mass | Low–Medium |
| Outdoor Adventure | Nature | Intimate | High |
| Night Club / Bar | Entertainment | Social | High |
| Workshop / Class | Cultural | Intimate | Medium |
| Networking Event | Cultural or Entertainment | Social | Medium |

Layer value domains:
- **Evidence:** Budget(Low/Med/High), GroupSize(Solo/Small/Large),
  ActivityLevel(Relaxed/Moderate/Active), Setting(Indoor/Outdoor/Either),
  TimeOfDay(Day/Evening/Night), Interest(Arts/Music/Food/Sports).
- **Latent:** VenueType(Cultural/Nature/Culinary [+Entertainment per catalogue]),
  SocialContext(Intimate/Social/Mass), EnergyProfile(Low/Medium/High).
- **Output:** EventRec — posterior over the 8 events; top three presented.

---

## 7. Robustness & safety

| Issue | Handling |
|---|---|
| Face not detected reliably | Short smoothing window before starting interaction |
| User gives unclear answers | One follow-up question, or use a default value |
| Some preferences missing | Run the BN with available evidence (partial) |
| Recommendation seems weak | Present top two or three events, not just one |
| Demo interruption | Return to Idle, wait for next user |

---

## 8. Evaluation criteria (demo checks)

Goal: a working prototype, not a full user study.

| Check | Expected result |
|---|---|
| Face detection | Pepper starts dialogue when a user is visible |
| Dialogue | Robot asks the main preference questions once |
| Parsing | User answers converted into Bayesian evidence values |
| Recommendation | Top events match collected preferences |
| Presentation | Pepper says recommendation + performs a simple gesture |

---

## 9. Timeline (from proposal)

| Week | Milestone |
|---|---|
| 1 | Set up qiBullet; webcam detection; Idle→Greeting |
| 2 | Dialogue manager, LLM evidence parser, Bayesian variables, initial CPTs |
| 3 | Connect inference, gesture library, recommendation presentation, safety filters |
| 4 | Optional speech input, face-recognition greeting, ranked explanations, demo tuning |

---

## 10. Status / next actions

- [x] Freeze `contracts.py` (Evidence/Result + label whitelists)
- [x] Scaffold `Project/src/` structure with stub interfaces per WP
- [x] WP1: FSM skeleton + Haar perception (scripted/webcam/pepper/hybrid sources)
- [x] WP4 (initial): real Pepper gestures + persistent TTS, coordinated with speech
- [x] WP2: LangGraph dialogue manager (Groq) — see below
- [x] WP3: 3-layer pyAgrum recommender + CPTs (`src/recommender/`, wired into main.py)
- [x] WP5: bridge the 3.8 robot FSM ↔ 3.11 dialogue service (JSON-over-stdio subprocess)
- [ ] WP5: demo tuning + evaluation-check pass on real hardware/sim

### WP5 integration (`src/dialogue_bridge.py` ↔ `dialogue/dialogue/bridge.py`)

The robot side (3.8) and dialogue (3.11) can't share an interpreter, so they run
as two processes connected by the bridge:

```
robot FSM (3.8)  --collect-->  dialogue service (3.11, LangGraph + Groq)
   ^  speaks question (WP4)  <--ask--          |
   |  sends typed/ASR answer  --answer-->       |
   '------ Evidence  <--evidence--  collect_evidence() finishes
```

`DialogueBridge` implements the `DialogueManager` contract on the robot side by
spawning the dialogue venv's `dialogue.bridge` and exchanging newline-delimited
JSON. For each `ask`, Pepper voices the question and the user answers; failures
fall back to empty evidence so the BN still runs. Enable with
`--dialogue real` (see [README](README.md)).

### WP2 as built (`Project/dialogue/`, standalone Python 3.11 uv project)

LLM = **Groq** (free Llama 3.3, OpenAI-compatible API; `GROQ_MODEL` overridable),
not pgmpy/xAI. Conversational, missing-aware questions are LLM-generated each turn.

Graph is a 6-node multi-agent flow (visible in LangGraph Studio):

```
Router -> QuestionFramer -> AnswerParser -> ConflictResolver -> Evaluator (-> Finish)
   ^___________________________________________|  reframe / conflict loops back
```

- **QuestionFramer** generates a fresh, open question anchored to the target slot.
- **AnswerParser** maps the reply to evidence (one answer may fill several slots);
  drops greeting/vague/catch-all guesses.
- **ConflictResolver** re-opens a slot and asks an either/or question when a new
  answer contradicts a decided one.
- **Evaluator** commits usable answers, reframes unusable ones (offering concrete
  options), and applies a default after N attempts.

Inputs: typed / speech (ASR) / scripted. Offline `ScriptedParser` + tests (3 pass).
Observability: **LangSmith tracing** (`LANGSMITH_*` in `src/.env`) and **LangGraph
Studio** (`uv run langgraph dev`). See [dialogue/RUNNING.md](dialogue/RUNNING.md).
