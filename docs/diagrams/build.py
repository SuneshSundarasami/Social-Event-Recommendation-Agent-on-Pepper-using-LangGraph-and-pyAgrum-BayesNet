"""Builds every README figure into docs/diagrams/*.svg.

Run from the repository root:

    uv run python docs/diagrams/build.py            # all figures
    uv run python docs/diagrams/build.py lifecycle  # just one

Each figure is one function below; edit its text or layout and re-run. The
drawing helpers live in svgkit.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svgkit import HALO, INK, LIGHT, MUTED, PALETTE, SURFACE, Svg, text_width  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
RED_FB = "#C62828"      # retry / failure / fallback arrows
GREEN_FB = "#1E8449"    # progress arrows
BLUE_FB = "#0A64B0"     # output to the visitor
GREY_ARC = "#7A8088"

FIGURES = []


def figure(fn):
    FIGURES.append(fn)
    return fn


def out(name: str) -> str:
    return os.path.join(OUT, name + ".svg")


def lines_block(s: Svg, cx, y, lines, size=12, gap=1.45, fill=INK):
    """Centred lines; a leading backtick renders that line in monospace."""
    for i, ln in enumerate(lines):
        mono = ln.startswith("`")
        s.text(cx, y + i * size * gap, ln.lstrip("`"), size=size - (0.5 if mono else 0),
               fill=fill, mono=mono)


def layer_label(s: Svg, y, label):
    s.line(30, y + 7, s.w - 30, y + 7, color="#C3C8CE", width=1)
    s.text(30, y, label, size=12.5, fill=MUTED, italic=True, anchor="start", halo=HALO)


# =========================================================================== main README


@figure
def architecture():
    s = Svg(1100, 560, "System architecture",
            "One route from perception to action, orchestrated by the interaction state machine "
            "across two Python processes")
    top = s.top

    kb = s.node(700, top, 300, 66, "Knowledge & data",
                lines=["event catalogue → CPTs · 8 event images", "schema in contracts.py"],
                color="grey", title_size=14, line_size=11.5)

    y, h, w, gap, x0 = top + 128, 136, 164, 24, 154
    xs = [x0 + i * (w + gap) for i in range(5)]
    visitor = s.oval(78, y + h / 2, 108, 74, "Visitor", "face · voice", outline=True)
    stages = [
        ("Perception", "sense + recognise",
         ["Haar + 5-frame smoothing", "LBPH face verification", "Whisper ASR (local)"], "red",
         "3.8 · 3.11"),
        ("Interpretation", "understand",
         ["Groq LLM parses replies", "into 6 discrete slots", "whitelist-validated"], "darkred",
         "3.11"),
        ("Dialogue manager", "manage turns",
         ["LangGraph slot-filling", "runs in the FSM's", "Conversation state"], "green", "3.11"),
        ("Bayesian reasoning", "decide",
         ["pyAgrum, 3 layers", "6 → 3 → 8 events", "ranked + explained"], "darkgreen", "3.8"),
        ("Behaviour & acting", "create + perform",
         ["neural TTS + gestures", "tablet image", "Pepper in qiBullet"], "blue", "3.8"),
    ]
    boxes = []
    for x, (title, role, lines, color, rt) in zip(xs, stages):
        boxes.append(s.node(x, y, w, h, title, role, lines, color, title_size=15, line_size=12))
        s.badge(x + w - 34, y, "Py " + rt, color="navy")

    s.edge([visitor.r(), boxes[0].l()], color=INK)
    for a, b in zip(boxes, boxes[1:]):
        s.edge([a.r(), b.l()], color=INK)

    yl = y - 30
    s.edge([boxes[2].t(-30), (boxes[2].cx - 30, yl), (boxes[0].cx, yl), boxes[0].t()],
           color=MUTED, dash=True, width=1.6,
           label="per turn: Pepper asks (TTS) → visitor answers (ASR) → until all slots are filled",
           lpos=((boxes[0].cx + boxes[2].cx - 30) / 2, yl - 8), lsize=11.5, litalic=True)
    s.edge([kb.b(-60), (kb.cx - 60, y - 22), (boxes[3].cx, y - 22), boxes[3].t()],
           color=GREY_ARC, width=1.6)
    s.edge([kb.b(80), (kb.cx + 80, y - 22), (boxes[4].cx, y - 22), boxes[4].t()],
           color=GREY_ARC, width=1.6)

    sy = y + h + 34
    s.rect(xs[0], sy, xs[-1] + w - xs[0], 42, s.fill("navy"), PALETTE["navy"][2], rx=21,
           shadow=True)
    s.text((xs[0] + xs[-1] + w) / 2, sy + 26,
           "Interaction FSM  ·  Idle → Greeting → Conversation → Reasoning → Recommendation → Farewell",
           size=13.5, weight="bold", fill="#FFFFFF")
    for b in boxes:
        s.line(b.cx, b.y + b.h, b.cx, sy, color="#9FB2C6", width=2, dash="3 4")

    ry = sy + 78
    s.edge([boxes[4].r(), (1070, boxes[4].cy), (1070, ry), (visitor.cx, ry), visitor.b()],
           color=BLUE_FB, width=2.2,
           label="the visitor sees and hears the result: speech · gestures · tablet image",
           lpos=(560, ry - 9), lsize=12)

    s.legend(28, s.h - 26, [("red", "sensing"), ("darkred", "understanding"),
                             ("green", "dialogue"), ("darkgreen", "reasoning"),
                             ("blue", "acting"), ("grey", "knowledge"), ("navy", "orchestration")])
    s.text(s.w - 28, s.h - 24, "Py 3.8 = robot process · Py 3.11 = dialogue process",
           size=11.5, fill=MUTED, anchor="end", italic=True)
    s.save(out("architecture"))


@figure
def lifecycle():
    s = Svg(1000, 500, "Interaction lifecycle",
            "One encounter, from the first stable face to the goodbye; "
            "the robot returns to Idle whenever the visitor walks away")
    w, h = 220, 104
    cols = [80, 390, 700]
    r1 = s.top + 60
    r2 = r1 + h + 92
    cards = [
        ("Idle", ["watches the camera until", "a face is stable"], "grey", cols[0], r1),
        ("Greeting", ["verifies the face (or asks consent)", "wave → nod → open arms"], "blue",
         cols[1], r1),
        ("Conversation", ["LangGraph asks open questions", "with a short 'talk' gesture"],
         "green", cols[2], r1),
        ("Reasoning", ["thinking pose while the", "Bayesian network ranks events"], "darkgreen",
         cols[2], r2),
        ("Recommendation", ["top 3 events, tablet image", "and a spoken explanation"], "blue",
         cols[1], r2),
        ("Farewell", ["goodbye wave, tablet cleared,", "state reset"], "navy", cols[0], r2),
    ]
    b = []
    for i, (title, lines, color, x, y) in enumerate(cards, 1):
        b.append(s.node(x, y, w, h, title, None, lines, color, title_size=16, line_size=12.5))
        s.number(x + 4, y + 4, i, color=color, r=13)

    s.raw('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (b[0].x - 50, b[0].cy, INK))
    s.edge([(b[0].x - 46, b[0].cy), b[0].l()], color=INK)

    def between(a, c):
        return (min(a.x + a.w, c.x + c.w) + max(a.x, c.x)) / 2

    s.edge([b[0].r(), b[1].l()], color=INK, label="face stable",
           lpos=(between(b[0], b[1]), b[0].cy - 10))
    s.text(between(b[0], b[1]), b[0].cy + 22, "4 of 5 frames", size=11, fill=MUTED)
    s.edge([b[1].r(), b[2].l()], color=INK, label="still present",
           lpos=(between(b[1], b[2]), b[1].cy - 10))
    s.edge([b[2].b(), b[3].t()], color=INK, label="evidence collected",
           lpos=(b[2].cx + 12, (r1 + h + r2) / 2 + 4), lanchor="start")
    s.edge([b[3].l(), b[4].r()], color=INK, label="events ranked",
           lpos=(between(b[3], b[4]), b[3].cy - 10))
    s.edge([b[4].l(), b[5].r()], color=INK, label="acknowledged",
           lpos=(between(b[4], b[5]), b[4].cy - 10))
    s.text(between(b[4], b[5]), b[4].cy + 22, "or timeout", size=11, fill=MUTED)
    s.edge([b[5].t(), b[0].b()], color=GREEN_FB, width=2.2, label="next visitor",
           lpos=(b[5].cx + 12, (r1 + h + r2) / 2 + 4), lanchor="start")

    s.edge([b[1].t(-40), (b[1].cx - 40, r1 - 20), (b[0].cx + 24, r1 - 20), b[0].t(24)],
           color=RED_FB, dash=True, width=1.6)
    s.edge([b[2].t(), (b[2].cx, r1 - 38), (b[0].cx - 24, r1 - 38), b[0].t(-24)],
           color=RED_FB, dash=True, width=1.6)
    s.text((b[0].cx + b[2].cx) / 2, r1 - 46, "visitor walked away → back to Idle", size=12,
           fill=RED_FB, italic=True, halo=HALO)

    s.text(s.w / 2, (r1 + h + r2) / 2 + 5, "InteractionFSM · src/state_machine.py",
           size=12, fill=MUTED, italic=True)
    s.save(out("lifecycle"))


@figure
def bayesian_network():
    s = Svg(1000, 650, "Bayesian recommendation network",
            "Each pair of answers drives one interpretable factor; the three factors rank the "
            "eight events (pyAgrum, exact inference)")
    top = s.top
    layer_label(s, top + 4, "Layer 1 · Evidence, gathered in the conversation (any subset)")
    ev = [("Interest", "Arts · Music", "Food · Sports"), ("Setting", "Indoor · Outdoor", "Either"),
          ("Group size", "Solo · Small", "Large"), ("Budget", "Low · Med · High", None),
          ("Activity level", "Relaxed · Moderate", "Active"),
          ("Time of day", "Day · Evening", "Night")]
    w, h, gap = 140, 76, 20
    x0 = (s.w - (6 * w + 5 * gap)) / 2
    y1 = top + 26
    eb = [s.node(x0 + i * (w + gap), y1, w, h, t, None, [a] + ([b] if b else []), "steel",
                 title_size=15, line_size=12) for i, (t, a, b) in enumerate(ev)]

    y2l = y1 + h + 70
    layer_label(s, y2l, "Layer 2 · Latent factors")
    y2 = y2l + 22
    lat = [("Venue type", ["Cultural · Entertainment", "Nature · Culinary"]),
           ("Social context", ["Intimate · Social · Mass"]),
           ("Energy profile", ["Low · Medium · High"])]
    lb = []
    for i, (t, lines) in enumerate(lat):
        cx = (eb[2 * i].cx + eb[2 * i + 1].cx) / 2
        lb.append(s.node(cx - 115, y2, 230, 76, t, None, lines, "green", title_size=15.5,
                         line_size=12))
    for i, e in enumerate(eb):
        left = i % 2 == 0
        tx, ty = lb[i // 2].t(-34 if left else 34)
        s.edge([e.b(), (tx, ty)], color=GREY_ARC, width=1.8, r=0)
        px, py = e.cx + (tx - e.cx) * 0.3, e.y + h + (ty - e.y - h) * 0.3
        s.text(px + (-12 if left else 12), py + 4, "base" if left else "× modifier", size=11,
               fill=MUTED, italic=True, anchor="end" if left else "start", halo=HALO)

    y3l = y2 + 76 + 46
    layer_label(s, y3l, "Layer 3 · Recommendation")
    y3 = y3l + 22
    rec = s.node(s.w / 2 - 170, y3, 340, 72, "Event recommendation",
                 lines=["ranked posterior over the eight events"], color="salmon",
                 title_size=16.5, line_size=12.5)
    for i, l in enumerate(lb):
        s.edge([l.b(), rec.t((i - 1) * 90)], color=GREY_ARC, width=1.8, r=0)

    events = ["Museum", "Concert", "Sports", "Food", "Outdoor", "Nightlife", "Workshop",
              "Networking"]
    cw, cg = 104, 14
    cx0 = (s.w - (8 * cw + 7 * cg)) / 2
    cy = y3 + 72 + 46
    for i in range(len(events)):
        s.edge([rec.b(), (cx0 + i * (cw + cg) + cw / 2, cy)], color="#E3A869", width=1.3,
               head=False, r=0)
    for i, e in enumerate(events):
        s.outline(cx0 + i * (cw + cg), cy, cw, 34, e, color="orange", title_size=13.5, rx=8)
    s.save(out("bayesian-network"))


# =========================================================================== src/README.md


@figure
def contracts():
    s = Svg(1000, 660, "Component contracts",
            "The state machine talks to every component through four small Protocol interfaces, "
            "so any part can be swapped for a stub")
    top = s.top
    fsm = s.hexagon(500, top + 34, 320, 64, "InteractionFSM", ["src/state_machine.py"])
    cols = [
        ("Perception", "red",
         ["tick() → bool", "face_stable() → bool", "is_present() → bool", "reset()", "release()"],
         [("HaarFaceDetector", False), ("ThreadedPerception", False),
          ("PreviewPerception", False), ("ScriptedPerception", True)]),
        ("DialogueManager", "green", ["collect_evidence()", "  → Evidence"],
         [("DialogueBridge", False), ("StubDialogueManager", True)]),
        ("Recommender", "darkgreen",
         ["recommend(evidence)", "  → Result", "explain(event,", "        evidence) → str"],
         [("BayesianRecommender", False), ("StubRecommender", True)]),
        ("Behaviour", "blue",
         ["say(text)", "gesture(name)", "say_with_gesture(", "    text, name)", "present(result)"],
         [("PepperBehaviour", False), ("ConsoleBehaviour", True)]),
    ]
    w, gap, x0 = 220, 20, 30
    yh = top + 120
    body_h = 132
    for i, (name, color, methods, impls) in enumerate(cols):
        x = x0 + i * (w + gap)
        cx = x + w / 2
        fill, stroke, tcol = LIGHT[color]
        s.rect(x, yh + 30, w, body_h + 28, SURFACE, stroke, rx=12, sw=1.6, shadow=True)
        head = s.node(x, yh, w, 58, name, "«Protocol»", (), color, title_size=15.5)
        for j, m in enumerate(methods):
            s.text(x + 18, yh + 84 + j * 20, m, size=12.5, mono=True, fill=INK, anchor="start")
        off = (i - 1.5) * 34
        s.edge([fsm.b(off), (fsm.cx + off, yh - 28), (cx, yh - 28), head.t()], color=INK,
               width=1.8)
        yc = yh + 30 + body_h + 28 + 30
        s.text(cx, yc, "implemented by", size=11.5, fill=MUTED, italic=True)
        for k, (impl, stub) in enumerate(impls):
            s.outline(x + 10, yc + 12 + k * 40, w - 20, 32, impl, color=color, title_size=12.5,
                      rx=8, dash="5 4" if stub else None)
    ly = s.h - 26
    s.rect(30, ly - 11, 34, 16, SURFACE, "#8C959F", rx=5, sw=1.4, dash="5 4")
    s.text(72, ly + 1.5, "dashed = stub, used for tests and as the fallback when the real "
                         "component can't start", size=12, fill=MUTED, anchor="start")
    s.save(out("contracts"))


@figure
def fsm():
    s = Svg(1000, 610, "InteractionFSM in detail",
            "Each state handler returns the next state; the code names on the arrows are the "
            "checks that drive each transition")
    r1 = s.top + 62
    h = 118
    idle = s.node(40, r1, 150, h, "Idle", None, ["polls perception", "every 0.1 s"], "grey",
                  title_size=16)
    gc = s.container(290, r1 - 6, 410, h + 12, "Greeting", None, "blue", rx=16, label_size=15)
    steps = [("verify /", "enrol", "red"), ("wave", None, "blue"), ("nod", None, "blue"),
             ("open", "arms", "blue")]
    sx = [310 + k * 98 for k in range(4)]
    sb = []
    for x, (a, b2, color) in zip(sx, steps):
        sb.append(s.node(x, r1 + 40, 80, 50, a, None, [b2] if b2 else [], color, rx=10,
                         title_size=13, line_size=13))
    for a, c in zip(sb, sb[1:]):
        s.edge([a.r(), c.l()], color=INK, width=1.6)
    s.text(gc.cx, r1 + h - 2, "_identify_or_enroll(), then three say_with_gesture() beats",
           size=11, fill=MUTED, italic=True)
    conv = s.node(780, r1, 190, h, "Conversation", None,
                  ["dialogue.collect_evidence()", "then validate_evidence()"],
                  "green", title_size=16, line_size=11.5)

    r2 = r1 + h + 100
    reas = s.node(780, r2, 190, h, "Reasoning", None, ["'Let me think about that.'",
                                                      "+ think gesture"], "darkgreen",
                  title_size=16)
    reco = s.node(420, r2, 230, h, "Recommendation", None,
                  ["present gesture · top 3", "tablet image · explain()", "wait for ack"],
                  "blue", title_size=16)
    fare = s.node(40, r2, 150, h, "Farewell", None, ["goodbye + wave", "reset everything"],
                  "navy", title_size=16)

    s.raw('<circle cx="22" cy="%.1f" r="6" fill="%s"/>' % (idle.cy, INK))
    s.edge([(26, idle.cy), idle.l()], color=INK)
    s.edge([idle.r(), (gc.x, idle.cy)], color=INK, label="face_stable()", lmono=True,
           lpos=((idle.x + idle.w + gc.x) / 2, idle.cy - 10), lsize=11.5)
    s.edge([(gc.x + gc.w, conv.cy), conv.l()], color=INK, label="still present", lmono=False,
           lpos=((gc.x + gc.w + conv.x) / 2, conv.cy - 10), lsize=11.5)
    s.edge([conv.b(), reas.t()], color=INK, label="evidence validated",
           lpos=(conv.cx + 12, (r1 + h + r2) / 2 + 4), lanchor="start")
    s.edge([reas.l(), reco.r()], color=INK, label="recommend()", lmono=True,
           lpos=((reco.x + reco.w + reas.x) / 2, reas.cy - 10), lsize=11.5)
    s.edge([reco.l(), fare.r()], color=INK, label="visitor leaves",
           lpos=((fare.x + fare.w + reco.x) / 2, reco.cy - 10), lsize=11.5)
    s.text((fare.x + fare.w + reco.x) / 2, reco.cy + 22, "or 15 s timeout", size=11,
           fill=MUTED)
    yb = r2 + h + 40
    s.edge([reco.b(), (reco.cx, yb), (fare.cx, yb), fare.b()], color="#C98A0C", dash=True,
           width=1.8, label="empty result → \"I'm sorry, I couldn't find a good match right now.\"",
           lpos=((reco.cx + fare.cx) / 2 + 20, yb - 8), lsize=11.5, litalic=True)
    s.edge([fare.t(), idle.b()], color=GREEN_FB, width=2.2, label="back to Idle",
           lpos=(fare.cx + 12, (r1 + h + r2) / 2 + 4), lanchor="start")

    s.edge([(gc.cx - 60, gc.y), (gc.cx - 60, r1 - 26), (idle.cx + 24, r1 - 26), idle.t(24)],
           color=RED_FB, dash=True, width=1.6)
    s.edge([conv.t(), (conv.cx, r1 - 44), (idle.cx - 24, r1 - 44), idle.t(-24)],
           color=RED_FB, dash=True, width=1.6)
    s.text((idle.cx + conv.cx) / 2, r1 - 52, "_still_present() is false → visitor left → Idle",
           size=12, fill=RED_FB, italic=True, halo=HALO)
    s.save(out("fsm"))


@figure
def run_modes():
    s = Svg(1000, 760, "Run modes",
            "How src/main.py assembles the system for each --source, with a fallback at every step")
    top = s.top
    s.pill(500, top + 16, "uv run python src/main.py --source …", color="navy", mono=True,
           size=13, h=36)
    layer_label(s, top + 56, "① --source decides where faces come from and whether Pepper is spawned")
    yc = top + 78
    cards = [
        ("scripted", "grey", ["ScriptedPerception", "ConsoleBehaviour", "", "no camera, no simulator"]),
        ("webcam", "red", ["`HaarFaceDetector(webcam)", "`+ FaceRecognizer", "`ConsoleBehaviour",
                           "local webcam, no simulator"]),
        ("pepper", "blue", ["`HaarFaceDetector(Pepper cam)", "`+ FaceRecognizer",
                            "`PepperBehaviour", "Pepper spawned in qiBullet"]),
        ("hybrid", "green", ["`ThreadedPerception(webcam)", "`+ FaceRecognizer + preview",
                             "`PepperBehaviour", "Pepper spawned in qiBullet"]),
    ]
    w, gap, x0 = 220, 20, 30
    bottoms = []
    for i, (name, color, lines) in enumerate(cards):
        x = x0 + i * (w + gap)
        fill, stroke, _ = LIGHT[color]
        s.rect(x, yc + 26, w, 126, SURFACE, stroke, rx=12, sw=1.6, shadow=True)
        s.node(x, yc, w, 46, "--source " + name, None, (), color, title_size=14.5)
        rows = [ln for ln in lines]
        for j, ln in enumerate(rows):
            if not ln:
                continue
            italic = not ln.startswith("`") and j == 3
            mono = ln.startswith("`") or (j < 3 and name == "scripted")
            s.text(x + w / 2, yc + 76 + j * 21, ln.lstrip("`"), size=12 if not mono else 11.8,
                   mono=mono, fill=MUTED if italic else INK, italic=italic)
        bottoms.append((x + w / 2, yc + 152))
    s.badge(x0 + 3 * (w + gap) + w - 52, yc, "demo setup", color="amber")
    ybus = yc + 182
    for bx, by in bottoms:
        s.line(bx, by, bx, ybus, color=INK, width=1.8)
    s.line(bottoms[0][0], ybus, bottoms[-1][0], ybus, color=INK, width=1.8)

    layer_label(s, ybus + 30, "② --dialogue")
    yd = ybus + 60
    stub = s.node(250, yd, 210, 64, "StubDialogueManager", None, ["console stand-in"], "grey",
                  title_size=14)
    real = s.node(540, yd, 210, 64, "DialogueBridge", None, ["LangGraph service · Py 3.11"],
                  "green", title_size=14)
    s.edge([(500, ybus), (500, ybus + 18), (stub.cx, ybus + 18), stub.t()], color=INK,
           label="stub", lmono=True, lpos=(stub.cx - 10, yd - 12), lanchor="end")
    s.edge([(500, ybus + 18), (real.cx, ybus + 18), real.t()], color=INK, label="real",
           lmono=True, lpos=(real.cx + 10, yd - 12), lanchor="start")
    s.edge([real.l(), stub.r()], color=RED_FB, dash=True, width=1.6, label="fails to start",
           lpos=(500, real.cy - 8), lsize=11.5, litalic=True)

    layer_label(s, yd + 64 + 28, "③ recommender")
    yr = yd + 64 + 58
    bn = s.node(395, yr, 210, 64, "BayesianRecommender", None, ["pyAgrum network"], "darkgreen",
                title_size=14)
    bstub = s.node(740, yr, 200, 64, "StubRecommender", None, ["toy deterministic ranker"],
                   "grey", title_size=14)
    s.edge([stub.b(), (stub.cx, yr - 18), (bn.cx - 30, yr - 18), bn.t(-30)], color=INK)
    s.edge([real.b(), (real.cx, yr - 18), (bn.cx + 30, yr - 18), bn.t(30)], color=INK)
    s.edge([bn.r(), bstub.l()], color=RED_FB, dash=True, width=1.6,
           label="pyAgrum unavailable", lpos=((bn.x + bn.w + bstub.x) / 2, bn.cy - 8),
           lsize=11.5, litalic=True)

    layer_label(s, yr + 64 + 28, "④ run")
    run = s.pill(500, yr + 64 + 76, "InteractionFSM.run()", color="navy", mono=True, size=13.5,
                 h=38)
    s.edge([bn.b(), run.t()], color=INK)
    s.edge([bstub.b(), (bstub.cx, run.cy), run.r()], color=GREY_ARC, dash=True, width=1.6)
    s.save(out("run-modes"))


@figure
def bridge_sequence():
    s = Svg(1000, 950, "The two-process bridge",
            "One evidence-collection run over newline-delimited JSON on the dialogue "
            "process's stdin and stdout")
    top = s.top
    H = s.h
    s.container(20, top, 528, H - top - 20, "Robot process · Python 3.8", None, "blue", rx=18,
                label_size=13.5)
    s.container(560, top, 420, H - top - 20, "Dialogue process · Python 3.11", None, "green",
                rx=18, label_size=13.5)
    lanes = {"pepper": 104, "fsm": 270, "client": 440, "server": 680, "graph": 880}
    heads = [("pepper", "Pepper", "voice + gestures", "blue"),
             ("fsm", "InteractionFSM", "state_machine.py", "navy"),
             ("client", "DialogueBridge", "dialogue_bridge.py", "grey"),
             ("server", "dialogue.bridge", "bridge.py", "grey"),
             ("graph", "LangGraph", "graph.py", "green")]
    hy = top + 46
    for key, title, role, color in heads:
        cx = lanes[key]
        s.line(cx, hy + 50, cx, H - 40, color="#AEB4BB", width=1.4, dash="4 4")
        s.node(cx - 74, hy, 148, 50, title, role, (), color, title_size=13.5, line_size=11)

    y = hy + 92
    n = [0]

    def msg(a, b, label, reply=False, mono=True, color=None):
        nonlocal y
        n[0] += 1
        xa, xb = lanes[a], lanes[b]
        c = color or (MUTED if reply else INK)
        s.edge([(xa, y), (xb + (-6 if xb > xa else 6), y)], color=c, width=1.7,
               dash="6 4" if reply else False)
        s.text((xa + xb) / 2, y - 8, label, size=11.8, fill=c if reply else INK, mono=mono,
               halo=HALO)
        s.number(xa, y, n[0], color="navy", r=10)
        y += 40

    msg("client", "server", "spawn  python -m dialogue.bridge", mono=True)
    msg("server", "client", '{"event": "ready"}', reply=True)
    msg("fsm", "client", "collect_evidence()")
    msg("client", "server", '{"cmd": "collect"}')
    msg("server", "graph", "invoke graph")
    loop_top = y - 18
    y += 14
    msg("graph", "server", "ask(question)")
    msg("server", "client", '{"event": "ask", "question": …}', reply=True)
    msg("client", "pepper", 'say_with_gesture(question, "talk")', color=BLUE_FB)
    alt_top = y - 18
    y += 14
    msg("client", "server", '{"answer": "…"}  typed reply')
    s.line(380, y - 14, 780, y - 14, color="#B9BFC6", width=1.2, dash="5 4")
    s.text(392, y + 2, "[spoken]", size=11, fill=MUTED, italic=True, anchor="start")
    y += 12
    msg("client", "server", '{"spoken": true}')
    s.rect(lanes["server"] - 78, y - 20, 156, 30, LIGHT["amber"][0], LIGHT["amber"][1], rx=8,
           sw=1.4)
    s.text(lanes["server"], y - 1, "record mic → Whisper", size=11.5, fill="#6B4E00")
    y += 34
    msg("server", "client", '{"event": "heard", "text": …}', reply=True)
    s.frame(380, alt_top, 400, y - alt_top - 18, "alt", "[typed]")
    msg("server", "graph", "reply")
    s.frame(36, loop_top, 930, y - loop_top - 14, "loop", "for each question, until every slot "
                                                         "is filled or defaulted")
    y += 10
    msg("graph", "server", "final evidence", reply=True)
    msg("server", "client", '{"event": "evidence", …}', reply=True)
    msg("client", "fsm", "Evidence", reply=True)
    s.save(out("bridge-sequence"))


# =========================================================================== perception


@figure
def perception_pipeline():
    s = Svg(1000, 440, "Perception pipeline",
            "Detection says someone is there; recognition says who it is. Everything runs on "
            "the device.")
    top = s.top
    cam1 = s.outline(30, top + 34, 170, 58, "Webcam", ["DirectShow first on Windows"],
                     color="red", title_size=13.5, line_size=11.5)
    cam2 = s.outline(30, top + 112, 170, 58, "Pepper head camera", ["qiBullet · QVGA"],
                     color="red", title_size=13.5, line_size=11.5)
    haar = s.node(250, top + 46, 180, 112, "Haar cascade", "detect",
                  ["grayscale frame", "scale 1.1 · 5 neighbours", "faces ≥ 60 px"], "red",
                  line_size=12)
    s.edge([cam1.r(), (225, cam1.cy), (225, haar.cy - 10), haar.l(-10)], color=INK, width=1.8)
    s.edge([cam2.r(), (225, cam2.cy), (225, haar.cy + 10), haar.l(10)], color=INK, width=1.8)

    win = s.outline(480, top + 26, 236, 104, None, (), color="green")
    s.text(win.cx, win.y + 24, "sliding window · last 5 frames", size=12.5, weight="bold",
           fill=LIGHT["green"][2])
    for k, ok in enumerate([True, True, False, True, True]):
        s.tile(498 + k * 42, top + 72, ok, size=34)
    s.edge([haar.r(-24), (455, haar.cy - 24), (455, win.cy), win.l()], color=INK, width=1.8)

    st = s.node(770, top + 4, 200, 66, "face_stable()", None, ["≥ 4 of 5, window full", "→ start the interaction"],
                "green", title_size=14, line_size=11.5)
    pr = s.node(770, top + 86, 200, 66, "is_present()", None, ["≥ 2 of 5", "→ keep it going"],
                "lightgreen", title_size=14, line_size=11.5)
    s.edge([win.r(-14), (742, win.cy - 14), (742, st.cy), st.l()], color=INK, width=1.8)
    s.edge([win.r(14), (742, win.cy + 14), (742, pr.cy), pr.l()], color=INK, width=1.8)

    lb = s.node(480, top + 200, 236, 90, "LBPH recognizer", "verify",
                ["largest face · 200×200 crop", "trained on known_faces/"], "darkred",
                line_size=12)
    s.edge([haar.b(), (haar.cx, lb.cy), lb.l()], color=INK, width=1.8, label="largest face box",
           lpos=((haar.cx + lb.x) / 2, lb.cy - 8), lsize=11.5)
    d = s.diamond(808, lb.cy, 120, 86, "distance\n≤ 75?", size=12.5)
    s.edge([lb.r(), d.l()], color=INK, width=1.8)
    yes = s.pill(918, top + 214, "name", color="green", size=13, h=32)
    no = s.pill(918, top + 278, "None", color="grey", size=13, h=32)
    s.edge([d.t(), (d.cx, yes.cy), yes.l()], color=GREEN_FB, width=1.8, label="yes",
           lpos=(d.cx + 30, yes.cy - 7), lsize=11.5)
    s.edge([d.b(), (d.cx, no.cy), no.l()], color=MUTED, width=1.8, label="no",
           lpos=(d.cx + 30, no.cy - 7), lsize=11.5)
    s.text(lb.cx + 60, lb.y + lb.h + 26, "distance ≈ 0 for a genuine match · ≈ 180 for a stranger",
           size=11.5, fill=MUTED, italic=True)
    s.save(out("perception-pipeline"))


@figure
def smoothing():
    s = Svg(1000, 380, "Two thresholds over one window",
            "A strict bar to start an interaction, a lenient one to keep it going")
    top = s.top
    hy = top + 8
    s.text(60, hy, "last five frames", size=12.5, weight="bold", fill=MUTED, anchor="start")
    for x, fn, rule in ((498, "face_stable()", "≥ 4 of 5"), (658, "is_present()", "≥ 2 of 5")):
        s.text(x, hy - 8, fn, size=12.5, weight="bold", fill=INK, mono=True)
        s.text(x, hy + 8, rule, size=12, fill=MUTED)
    s.text(820, hy, "effect", size=12.5, weight="bold", fill=MUTED)
    rows = [([1, 1, 0, 1, 1], True, True, "Idle → Greeting", "green"),
            ([0, 1, 0, 0, 1], False, True, "conversation continues", "blue"),
            ([0, 0, 0, 0, 1], False, False, "visitor left → Idle", "red")]
    for i, (pattern, stable, present, effect, color) in enumerate(rows):
        y = hy + 34 + i * 70
        s.rect(30, y - 8, s.w - 60, 58, "#F7F8FA" if i % 2 == 0 else "none", None, rx=10)
        for k, ok in enumerate(pattern):
            s.tile(60 + k * 46, y, bool(ok), size=40)
        s.text(60 + 5 * 46 + 6, y + 25, "%d / 5" % sum(pattern), size=13, fill=MUTED,
               anchor="start", weight="bold")
        s.tile(480, y + 2, stable, size=36)
        s.tile(640, y + 2, present, size=36)
        s.pill(820, y + 20, effect, color=color, size=13, h=34)
    s.save(out("smoothing"))


@figure
def perception_wrappers():
    s = Svg(1000, 380, "Perception wrappers",
            "Layers that add threading and a live preview without the state machine noticing")
    top = s.top
    fsm = s.node(30, top + 72, 150, 80, "InteractionFSM", None, ["polls tick()"], "navy",
                 title_size=14.5)
    outer = s.container(290, top + 10, 420, 206, "ThreadedPerception",
                        "own capture thread at 15 fps · lock-guarded cached state", "blue",
                        rx=20, label_size=14.5)
    inner = s.container(316, top + 74, 368, 124, "HaarFaceDetector",
                        "5-frame smoothing · optional FaceRecognizer", "red", rx=16,
                        label_size=13.5)
    src = s.node(415, top + 136, 170, 44, "WebcamSource", None, (), "red", title_size=13.5)
    s.edge([fsm.r(), (outer.x, fsm.cy)], color=INK, label="cheap cached reads",
           lpos=((fsm.x + fsm.w + outer.x) / 2, fsm.cy - 10), lsize=11)

    # preview window mock-up
    wx, wy, ww, wh = 760, top + 14, 210, 168
    s.rect(wx, wy, ww, wh, "#1E242B", "#0F1317", rx=10, shadow=True)
    s.rect(wx, wy, ww, 24, "#2E353D", None, rx=10)
    s.rect(wx, wy + 14, ww, 10, "#2E353D", None, rx=0)
    for k, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        s.raw('<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s"/>' % (wx + 14 + k * 14, wy + 12, c))
    s.text(wx + ww / 2 + 16, wy + 16, "preview", size=11, fill="#C9D1D9")
    s.raw('<circle cx="%.1f" cy="%.1f" r="26" fill="#5B6570"/>' % (wx + ww / 2, wy + 88))
    s.raw('<path d="M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f Z" fill="#5B6570"/>'
          % (wx + ww / 2 - 52, wy + wh - 2, wx + ww / 2, wy + 100, wx + ww / 2 + 52, wy + wh - 2))
    s.rect(wx + ww / 2 - 40, wy + 50, 80, 82, "none", "#34C759", rx=4, sw=2.5)
    s.text(wx + 12, wy + 44, "STABLE", size=12, weight="bold", fill="#34C759", anchor="start")
    s.edge([(outer.x + outer.w, fsm.cy), (wx, fsm.cy)], color=BLUE_FB, dash=True, width=1.6,
           label="draws", lpos=((outer.x + outer.w + wx) / 2, fsm.cy - 9), lsize=11)
    lx = wx
    for color, label in (("green", "STABLE"), ("amber", "present"), ("red", "searching")):
        wl = text_width(label, 11, "bold") + 20
        s.rect(lx, wy + wh + 16, wl, 24, s.fill(color), PALETTE[color][2], rx=12)
        s.text(lx + wl / 2, wy + wh + 32, label, size=11, weight="bold",
               fill=PALETTE[color][3])
        lx += wl + 8
    s.text(30, s.h - 24, "ScriptedPerception replaces all of this in --source scripted: the "
                         "visitor arrives at tick 3 and leaves at tick 12.",
           size=12, fill=MUTED, italic=True, anchor="start")
    s.save(out("perception-wrappers"))


@figure
def enrollment():
    s = Svg(1020, 470, "Face verification and consent-based enrollment",
            "Runs at the start of the Greeting state whenever a camera is in use")
    top = s.top
    ya, yb = top + 26, top + 126
    yc = yb + 136
    s.container(244, yb - 66, 556, yc - yb + 120, None, None, "amber", rx=18)
    start = s.pill(72, yb, "stable face", color="navy", size=13, h=34)
    d1 = s.diamond(180, yb, 104, 82, "known\nface?")
    b1 = s.bubble(276, yb - 32, 182, 58, ["I don't think we've met yet.", "What's your name?"],
                  size=12, tail="bottom-left")
    b2 = s.bubble(474, yb - 32, 196, 58, ["Would it be okay if I", "remember your face?"],
                  size=12, tail="bottom-left")
    d2 = s.diamond(730, yb, 104, 82, "yes?")
    cap = s.node(812, yb - 36, 178, 72, "Capture ~15 samples", None,
                 ["from the open camera feed", "then retrain() at once"], "red", title_size=13.5,
                 line_size=11.5)
    nothing = s.node(560, yc - 34, 180, 68, "Nothing captured", None,
                     ['"No problem, {name}."', "name used this time only"], "amber",
                     title_size=13.5, line_size=11.5)
    greet = s.node(812, yc - 36, 178, 72, "Greetings, {name}.", None,
                   ["personal greeting, then", "wave → nod → open arms"], "green",
                   title_size=13.5, line_size=11.5)

    s.edge([start.r(), d1.l()], color=INK)
    s.edge([d1.r(), b1.l()], color=INK, label="no", lpos=((d1.x + d1.w + b1.x) / 2, yb - 8),
           lsize=11.5)
    s.edge([b1.r(), b2.l()], color=INK, head=True)
    s.edge([b2.r(), d2.l()], color=INK)
    s.edge([d2.r(), cap.l()], color=GREEN_FB, label="yes", lpos=((d2.x + d2.w + cap.x) / 2, yb - 8),
           lsize=11.5)
    s.edge([d2.b(), (d2.cx, yc - 34)], color="#C98A0C", label="no", lpos=(d2.cx + 14, yb + 70),
           lanchor="start", lsize=11.5)
    s.edge([cap.b(), greet.t()], color=GREEN_FB)
    s.edge([nothing.r(), greet.l()], color=INK)
    s.edge([d1.t(), (d1.cx, ya), (1004, ya), (1004, greet.cy), greet.r()], color=GREEN_FB,
           width=2, label="yes · recognised by LBPH (distance ≤ 75) → greet by name, no questions",
           lpos=(600, ya - 8), lsize=12)
    s.text(272, yc - 6, "Consent gate", size=14, weight="bold", fill="#8A5A00", anchor="start")
    s.text(272, yc + 14, "face samples are stored only", size=12, fill=MUTED, anchor="start")
    s.text(272, yc + 31, "after an explicit yes", size=12, fill=MUTED, anchor="start")
    s.text(30, s.h - 24, "No camera wired (--source scripted) or no name given → generic "
                         "\"Greetings.\" without enrollment.", size=12, fill=MUTED, italic=True,
           anchor="start")
    s.save(out("enrollment"))


# =========================================================================== dialogue


@figure
def dialogue_graph():
    s = Svg(1240, 520, "Dialogue manager",
            "A LangGraph StateGraph (dialogue/graph.py) that runs inside the FSM's Conversation "
            "state and fills the six preference slots")
    top = s.top
    s.container(24, top, 1192, s.h - top - 24, None, None, "green", rx=22)
    y3, y2, y1 = top + 40, top + 66, top + 92
    yn, h = top + 128, 86
    start = s.oval(86, yn + h / 2, 84, 46, "START")
    xs = [(166, 150), (372, 170), (598, 170), (824, 165), (1045, 150)]
    specs = [("Router", ["pick a random", "unfilled slot"]),
             ("QuestionFramer", ["LLM writes one open", "question · robot asks"]),
             ("AnswerParser", ["abuse filter → LLM", "→ whitelist → guard"]),
             ("ConflictResolver", ["contradiction?", "re-open the slot"]),
             ("Evaluator", ["commit, rephrase", "or apply default"])]
    n = [s.node(x, yn, w, h, t, None, lines, "green", title_size=14.5, line_size=12)
         for (x, w), (t, lines) in zip(xs, specs)]
    router, qf, ap, cr, ev = n
    s.badge(qf.x + qf.w - 24, yn, "LLM", color="darkred")
    s.badge(ap.x + ap.w - 24, yn, "LLM", color="darkred")

    s.edge([start.r(), router.l()], color=INK)
    s.edge([router.r(), qf.l()], color=INK, label="missing", lpos=((router.x + router.w + qf.x) / 2, yn + h / 2 - 8), lsize=11)
    s.edge([qf.r(), ap.l()], color=INK)
    s.edge([ap.r(), cr.l()], color=INK)
    s.edge([cr.r(), ev.l()], color=INK, label="no conflict",
           lpos=((cr.x + cr.w + ev.x) / 2, yn + h / 2 - 8), lsize=10.5)

    s.edge([ev.t(35), (ev.cx + 35, y3), (router.cx, y3), router.t()], color=GREEN_FB, width=2,
           label="usable · committed · or default applied → next slot",
           lpos=((router.cx + ev.cx) / 2, y3 - 8), lsize=12)
    s.edge([ev.t(-25), (ev.cx - 25, y2), (qf.cx - 25, y2), qf.t(-25)], color=RED_FB, width=2,
           label="answer not usable → rephrase (up to 2×)", lpos=((qf.cx + ev.cx) / 2 + 60, y2 - 8),
           lsize=12)
    s.edge([cr.t(), (cr.cx, y1), (qf.cx + 25, y1), qf.t(25)], color=RED_FB, width=2,
           label="conflict → re-open slot, ask either/or", lpos=((qf.cx + cr.cx) / 2 + 10, y1 - 8),
           lsize=12)

    yf = yn + h + 58
    fin = s.node(router.x, yf, router.w, 48, "Finish", None, (), "lightgreen", title_size=14.5)
    end = s.oval(86, yf + 24, 84, 46, "END")
    s.edge([router.b(), fin.t()], color=INK, label="all filled",
           lpos=(router.cx + 10, yn + h + 32), lanchor="start", lsize=11)
    s.edge([fin.l(), end.r()], color=INK)
    s.edge([fin.r(), (1180, fin.cy)], color=GREEN_FB, dash=True, width=1.8,
           label="returns the Evidence dict to the FSM → Bayesian reasoning",
           lpos=(800, fin.cy - 8), lsize=12, litalic=True)
    s.text(qf.cx, yn + h + 22, "↑ question out via TTS · answer in via ASR or keyboard",
           size=11.5, fill=BLUE_FB, italic=True)
    s.text(ap.cx + 60, yn + h + 40, "↑ Groq LLM wrapped in safety guards", size=11.5,
           fill="#B32B21", italic=True)
    s.save(out("dialogue-graph"))


@figure
def safety_chain():
    s = Svg(1000, 520, "From free text to safe evidence",
            "Every reply passes the same chain of checks before it can influence a recommendation")
    top = s.top
    y, h = top + 30, 88
    reply = s.bubble(30, y + 6, 132, 72, ['"somewhere cheap,', 'outside — great!"'], size=12,
                     italic=True, title="VISITOR")
    steps = [(190, 138, "Abuse filter", None, ["flagged-word", "check"], "amber"),
             (352, 150, "LLM parser", "Groq", ["JSON mode", "temperature 0"], "darkred"),
             (526, 138, "Whitelist", None, ["legal slots and", "values only"], "amber"),
             (688, 138, "Weak-value guard", None, ["drop catch-alls", "without a cue"], "amber")]
    boxes = []
    for i, (x, w, t, role, lines, color) in enumerate(steps, 1):
        boxes.append(s.node(x, y, w, h, t, role, lines, color, title_size=14, line_size=12))
        s.number(x + 2, y + 2, i, color="navy", r=12)
    ev = s.node(850, y, 120, h, "Evidence", None, ["Budget=Low", "Setting=Outdoor"],
                "green", title_size=14.5, line_size=11.5)
    s.edge([reply.r(), boxes[0].l()], color=INK)
    for a, b in zip(boxes, boxes[1:]):
        s.edge([a.r(), b.l()], color=INK)
    s.edge([boxes[-1].r(), ev.l()], color=INK)

    rejects = [(0, "declined", ['"Let\'s keep this', 'friendly…" re-ask'], "red"),
               (1, "omitted", ["vague replies like", '"sure" fill nothing'], "grey"),
               (2, "dropped", ["unknown slots or", "illegal values"], "red"),
               (3, "dropped", ['"great" never becomes', "Setting=Either"], "red")]
    ry = y + h + 50
    for i, title, lines, color in rejects:
        b = boxes[i]
        r = s.outline(b.x, ry, b.w, 62, title, lines, color=color, title_size=12.5,
                      line_size=11.5)
        s.edge([b.b(), r.t()], color=RED_FB if color == "red" else MUTED, dash=True, width=1.5)

    dy = ry + 62 + 44
    s.container(30, dy, 940, 112, "Defence in depth", "the same whitelist runs three times",
                "navy", rx=18, label_size=14.5)
    guards = [("schema.whitelist()", "dialogue process"),
              ("validate_evidence()", "InteractionFSM"),
              ("validate_evidence()", "BayesianRecommender")]
    gx = [300, 520, 740]
    gb = []
    for x, (t, sub) in zip(gx, guards):
        gb.append(s.outline(x, dy + 30, 196, 54, t, [sub], color="navy", title_size=13,
                            line_size=11.5))
    for a, b in zip(gb, gb[1:]):
        s.edge([a.r(), b.l()], color=INK, width=1.8)
    s.text(48, dy + 82, "the network only ever sees", size=12, fill=MUTED, anchor="start")
    s.text(48, dy + 98, "legal labels", size=12, fill=MUTED, anchor="start")
    s.save(out("safety-chain"))


@figure
def speech_input():
    s = Svg(1000, 320, "Speech input",
            "Spoken replies are transcribed on the device with faster-whisper; no audio leaves "
            "the machine")
    top = s.top
    cy = top + 96
    mic = s.node(30, cy - 36, 130, 72, "Microphone", None, ["WHISPER_MIC_INDEX"], "red",
                 title_size=14, line_size=10.5)
    sr = s.node(192, cy - 44, 190, 88, "SpeechRecognition", "endpointing",
                ["ambient calibration", "stops on silence"], "red", title_size=14, line_size=12)
    d = s.diamond(464, cy, 112, 84, "CUDA\nGPU?")
    gpu = s.node(560, top + 6, 230, 70, "Whisper large-v3", None,
                 ["int8_float16 · ≈ 2.2 GB VRAM", "leaves room for the simulator"], "darkred",
                 title_size=14, line_size=11.5)
    cpu = s.node(560, top + 118, 230, 70, "Whisper small", None, ["int8 on the CPU"], "darkred",
                 title_size=14, line_size=12)
    tr = s.pill(905, cy, "transcript", color="green", size=13.5, h=40)
    s.edge([mic.r(), sr.l()], color=INK)
    s.edge([sr.r(), d.l()], color=INK)
    s.edge([d.t(), (d.cx, gpu.cy), gpu.l()], color=INK, label="yes", lpos=(d.cx + 30, gpu.cy - 8),
           lsize=11.5)
    s.edge([d.b(), (d.cx, cpu.cy), cpu.l()], color=INK, label="no", lpos=(d.cx + 30, cpu.cy - 8),
           lsize=11.5)
    s.edge([gpu.b(40), cpu.t(40)], color=RED_FB, dash=True, width=1.6)
    s.text(gpu.cx + 30, (gpu.y + gpu.h + cpu.y) / 2 + 4, "GPU error → reload on CPU once",
           size=11.5, fill=RED_FB, italic=True, anchor="end", halo=HALO)
    s.edge([gpu.r(), (840, gpu.cy), (840, tr.cy - 8), (tr.x, tr.cy - 8)], color=INK)
    s.edge([cpu.r(), (840, cpu.cy), (840, tr.cy + 8), (tr.x, tr.cy + 8)], color=INK)
    s.text(tr.cx, tr.y + tr.h + 20, "→ AnswerParser", size=12, fill=MUTED, italic=True)
    s.save(out("speech-input"))


# =========================================================================== recommender


def bars(s: Svg, x, y, w, h, values, colors, scale, labels=None, ref=None, fmt="%.2f"):
    """A small vertical bar chart inside (x, y, w, h)."""
    n = len(values)
    bw, gap = 28, 14
    total = n * bw + (n - 1) * gap
    bx0 = x + (w - total) / 2
    base = y + h
    s.line(x + 8, base, x + w - 8, base, color="#9AA0A6", width=1.2)
    if ref is not None:
        ry = base - ref * scale
        s.line(x + 8, ry, x + w - 8, ry, color="#C9CED4", width=1, dash="4 3")
        s.text(x + w - 8, ry - 4, "× 1", size=10, fill=MUTED, anchor="end")
    for i, (v, c) in enumerate(zip(values, colors)):
        bh = max(2, v * scale)
        s.rect(bx0 + i * (bw + gap), base - bh, bw, bh, s.fill(c), PALETTE[c][2], rx=4, sw=1)
        s.text(bx0 + i * (bw + gap) + bw / 2, base - bh - 6, fmt % v, size=11, fill=INK,
               weight="bold")


@figure
def cpt_generation():
    s = Svg(1000, 420, "How a latent CPT column is generated",
            "Worked example: VenueType given Interest = Arts and Setting = Indoor")
    top = s.top
    colors = ["purple", "blue", "green", "amber"]
    panels = [("Base", "from Interest = Arts", [0.70, 0.15, 0.05, 0.10], 150, None, "%.2f"),
              ("Modifier", "from Setting = Indoor", [1.2, 1.1, 0.4, 1.1], 88, 1.0, "×%.1f"),
              ("Product", "element-wise", [0.84, 0.165, 0.02, 0.11], 150, None, "%.3f"),
              ("CPT column", "normalised to sum 1", [0.740, 0.145, 0.018, 0.097], 150, None,
               "%.3f")]
    pw, ow, x0 = 205, 40, 30
    for i, (title, sub, vals, scale, ref, fmt) in enumerate(panels):
        x = x0 + i * (pw + ow)
        last = i == 3
        fill, stroke, tcol = LIGHT["green" if last else "white"]
        s.rect(x, top + 6, pw, 222, SURFACE, stroke if last else "#D0D5DB", rx=14,
               sw=2 if last else 1.4, shadow=True)
        s.text(x + pw / 2, top + 32, title, size=15, weight="bold", fill=tcol if last else INK)
        s.text(x + pw / 2, top + 50, sub, size=11.5, fill=MUTED, italic=True)
        bars(s, x, top + 70, pw, 140, vals, colors, scale, ref=ref, fmt=fmt)
        if i < 3:
            s.text(x + pw + ow / 2, top + 128, ["×", "=", "÷"][i], size=28, weight="bold",
                   fill=MUTED)
            if i == 2:
                s.text(x + pw + ow / 2, top + 146, "sum", size=11, fill=MUTED)
    s.legend(x0 + 150, top + 262, [("purple", "Cultural"), ("blue", "Entertainment"),
                                    ("green", "Nature"), ("amber", "Culinary")], size=12.5)
    s.text(s.w / 2, s.h - 32, "The same rule fills every column of VenueType, SocialContext and "
                              "EnergyProfile; EventRec's 36 columns come from the event profiles.",
           size=12, fill=MUTED, italic=True)
    s.save(out("cpt-generation"))


@figure
def inference():
    s = Svg(1000, 610, "Inference and explanation",
            "Real output for a visitor who said they like art, are going alone and want "
            "something relaxed")
    top = s.top
    s.text(30, top + 10, "Evidence", size=13, weight="bold", fill=MUTED, anchor="start")
    known = ["Interest = Arts", "GroupSize = Solo", "ActivityLevel = Relaxed"]
    unknown = ["Budget = ?", "Setting = ?", "TimeOfDay = ?"]
    cy = top + 24
    chips = []
    for t in known:
        chips.append(s.outline(30, cy, 200, 32, t, color="blue", title_size=12.5, rx=8))
        cy += 40
    for t in unknown:
        s.outline(30, cy, 200, 32, t, color="grey", title_size=12.5, rx=8, dash="5 4")
        cy += 40
    s.text(130, cy + 12, "unknown → uniform prior", size=11.5, fill=MUTED, italic=True)

    mid = top + 140
    wl = s.node(270, mid - 34, 130, 68, "Whitelist", None, ["validate_evidence()"], "amber",
                title_size=14, line_size=10.5)
    lp = s.node(436, mid - 50, 160, 100, "LazyPropagation", "pyAgrum", ["exact inference"],
                "darkgreen", title_size=14.5)
    s.edge([(236, mid), wl.l()], color=INK)
    s.line(232, top + 40, 232, top + 24 + 40 * 6 - 24, color=INK, width=1.6)
    s.edge([wl.r(), lp.l()], color=INK)

    # posterior chart
    cx0, cy0 = 640, top + 6
    s.text(cx0, cy0 + 4, "recommend() → ranked posterior", size=13, weight="bold", fill=INK,
           anchor="start", mono=False)
    post = [("Museum", 40.6), ("Workshop", 19.2), ("Food", 11.0), ("Networking", 10.7),
            ("Outdoor", 6.7), ("Concert", 5.3), ("Nightlife", 3.5), ("Sports", 2.9)]
    for i, (e, p) in enumerate(post):
        y = cy0 + 22 + i * 27
        s.text(cx0 + 82, y + 14, e, size=12, fill=INK, anchor="end")
        color = "salmon" if i == 0 else "steel"
        s.rect(cx0 + 90, y + 2, p * 5.2, 18, s.fill(color), PALETTE[color][2], rx=4, sw=1)
        s.text(cx0 + 96 + p * 5.2, y + 15, "%.1f%%" % p, size=11.5, fill=INK, anchor="start",
               weight="bold" if i == 0 else "normal")
    s.edge([lp.r(-24), (612, lp.cy - 24), (612, cy0 + 120), (cx0 - 4, cy0 + 120)], color=INK,
           width=1.8)

    ey = top + 270
    s.text(cx0, ey, "explain() → most likely latent values", size=13, weight="bold", fill=INK,
           anchor="start")
    lat = ["VenueType = Cultural", "SocialContext = Intimate", "EnergyProfile = Low"]
    for i, t in enumerate(lat):
        s.outline(cx0 + (i % 2) * 168, ey + 14 + (i // 2) * 40, 160, 32, t, color="green",
                  title_size=11.5, rx=8)
    s.edge([lp.r(24), (612, lp.cy + 24), (612, ey + 30), (cx0 - 4, ey + 30)], color=INK,
           width=1.8)
    s.bubble(cx0 - 10, ey + 104, 340, 84,
             ["I suggest Museum because your preferences", "point to a cultural venue with a",
              "intimate vibe and a low-key feel."], size=12, title="JARVIS", tail="bottom-left")
    s.save(out("inference"))


# =========================================================================== behaviour


@figure
def speech_gesture():
    s = Svg(1000, 400, "Speech and gesture in step",
            "say_with_gesture() starts both together and returns only when both have finished "
            "(the three greeting beats)")
    top = s.top
    x0, pps = 210, 62.0
    lanes = [("InteractionFSM", "navy"), ("gesture thread", "blue"), ("TTS worker", "green")]
    ly = [top + 22, top + 86, top + 150]
    lh = 46
    for (name, color), y in zip(lanes, ly):
        s.rect(30, y, 160, lh, LIGHT[color if color != "navy" else "navy"][0],
               LIGHT[color if color != "navy" else "navy"][1], rx=10, sw=1.4)
        s.text(110, y + lh / 2 + 4.5, name, size=13, weight="bold",
               fill=LIGHT[color if color != "navy" else "navy"][2])
        s.line(x0, y + lh / 2, x0 + 12.4 * pps, y + lh / 2, color="#E3E6EA", width=1)

    hatch = ('<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)"><rect width="7" height="7" fill="#F4F5F7"/>'
             '<line x1="0" y1="0" x2="0" y2="7" stroke="#D0D5DB" stroke-width="2.5"/></pattern>')
    s.defs["hatch"] = hatch

    beats = [(0.0, "wave", 5.0, '"Greetings, Sunesh…"', 3.4),
             (5.0, "nod", 1.6, '"Systems nominal…"', 2.4),
             (7.4, "open_arms", 3.2, '"Scanning the grid…"', 3.6)]

    def X(t):
        return x0 + t * pps

    for i, (t0, g, gd, line, sd) in enumerate(beats):
        end = t0 + max(gd, sd)
        s.rect(X(t0) + 2, ly[0] + 6, (end - t0) * pps - 4, lh - 12, s.fill("navy"),
               PALETTE["navy"][2], rx=8)
        s.text((X(t0) + X(end)) / 2, ly[0] + lh / 2 + 4.5, "beat %d" % (i + 1), size=12,
               weight="bold", fill="#FFFFFF")
        s.rect(X(t0) + 2, ly[1] + 6, gd * pps - 4, lh - 12, s.fill("blue"), PALETTE["blue"][2],
               rx=8)
        s.text((X(t0) + X(t0 + gd)) / 2, ly[1] + lh / 2 + 4.5, "%s · %.1f s" % (g, gd), size=12,
               weight="bold", fill="#FFFFFF")
        s.rect(X(t0) + 2, ly[2] + 6, sd * pps - 4, lh - 12, s.fill("green"),
               PALETTE["green"][2], rx=8)
        s.text((X(t0) + X(t0 + sd)) / 2, ly[2] + lh / 2 + 4.5, line, size=12, fill="#FFFFFF",
               italic=True)
        for lane, dur in ((1, gd), (2, sd)):
            if dur < end - t0:
                s.rect(X(t0 + dur) + 2, ly[lane] + 10, (end - t0 - dur) * pps - 4, lh - 20,
                       "url(#hatch)", "#D0D5DB", rx=6, sw=1)
        s.line(X(end), ly[0] - 4, X(end), ly[2] + lh + 6, color=RED_FB, width=1.6, dash="5 4")
        s.text(X(end), ly[0] - 9, "both done", size=11, fill=RED_FB, italic=True)

    ay = ly[2] + lh + 24
    s.line(x0, ay, X(12), ay, color="#9AA0A6", width=1.2)
    for t in range(0, 13):
        s.line(X(t), ay - 4, X(t), ay + 4, color="#9AA0A6", width=1.2)
        s.text(X(t), ay + 18, "%d s" % t if t in (0, 12) else str(t), size=11, fill=MUTED)
    s.rect(30, ay + 32, 30, 14, "url(#hatch)", "#D0D5DB", rx=4, sw=1)
    s.text(68, ay + 43, "finished early, waiting for the other half · gesture lengths from "
                        "pepper_behaviour.py, speech lengths illustrative",
           size=11.5, fill=MUTED, anchor="start", italic=True)
    s.save(out("speech-gesture"))


@figure
def voice():
    s = Svg(1000, 410, "Voice",
            "TTS_BACKEND picks the speech engine; lines are spoken on a worker thread so "
            "gestures can run at the same time")
    top = s.top
    sel = s.pill(118, top + 120, "TTS_BACKEND", color="navy", mono=True, size=13.5, h=40)
    edge_n = s.node(330, top + 8, 230, 80, "edge-tts neural voice", None,
                    ["en-GB-RyanNeural", "rate −8% · pitch −5 Hz"], "blue", title_size=14,
                    line_size=12)
    py = s.node(330, top + 128, 230, 64, "pyttsx3", None, ["offline SAPI5 / eSpeak"], "steel",
                title_size=14, line_size=12)
    none = s.node(330, top + 226, 230, 52, "silent", None, ["text printed only"], "grey",
                  title_size=14, line_size=11.5)
    for tgt, lab in ((edge_n, "auto · edge"), (py, "pyttsx3"), (none, "none")):
        s.edge([sel.r(), (240, sel.cy), (240, tgt.cy), tgt.l()], color=INK, label=lab, lmono=True,
               lpos=(285, tgt.cy - 8), lsize=11.5)
    s.edge([edge_n.b(60), py.t(60)], color=RED_FB, dash=True, width=1.6)
    s.text(edge_n.cx + 48, (edge_n.y + edge_n.h + py.y) / 2 + 4,
           "offline or failing → this line falls back", size=11.5, fill=RED_FB, italic=True,
           anchor="end", halo=HALO)
    eng = s.node(700, top + 66, 270, 98, "TtsEngine", "worker thread",
                 ["queues each line, speaks it,", "signals when it has finished"], "navy",
                 title_size=15, line_size=12)
    s.edge([edge_n.r(), (640, edge_n.cy), (640, eng.cy - 14), eng.l(-14)], color=INK)
    s.edge([py.r(), (640, py.cy), (640, eng.cy + 14), eng.l(14)], color=INK)
    s.save(out("voice"))


@figure
def tablet_display():
    s = Svg(1000, 470, "Tablet display",
            "qiBullet has no tablet API, so the image is drawn on a textured panel that follows "
            "Pepper's Tablet_frame link")
    top = s.top
    cy = top + 70
    pres = s.pill(94, cy, "present(result)", color="navy", mono=True, size=12.5, h=36)
    look = s.node(186, cy - 37, 150, 74, "Image lookup", None, ["imgs/ + event name", ".jpg or .avif"],
                  "grey", title_size=14, line_size=11.5)
    pil = s.node(366, cy - 50, 190, 100, "Pillow + AVIF", "prepare texture",
                 ["centre-crop to 16:9", "2048×1152 · Lanczos", "unsharp mask"], "blue",
                 title_size=14.5, line_size=11.5)
    d = s.diamond(640, cy, 132, 90, "Tablet_frame\nlink?", size=12)
    panel = s.node(744, cy - 46, 226, 92, "Textured panel", "on the tablet",
                   ["follows the link at ~33 Hz", "1.8 cm in front of the screen"], "blue",
                   title_size=14.5, line_size=11.5)
    win = s.node(560, cy + 120, 160, 64, "OpenCV window", None, ["desktop fallback"], "grey",
                 title_size=14, line_size=11.5)
    s.edge([pres.r(), look.l()], color=INK)
    s.edge([look.r(), pil.l()], color=INK)
    s.edge([pil.r(), d.l()], color=INK)
    s.edge([d.r(), panel.l()], color=GREEN_FB, label="yes", lpos=(724, cy - 8), lsize=11.5)
    s.edge([d.b(), win.t()], color=MUTED, label="no", lpos=(d.cx + 10, d.y + d.h + 18),
           lanchor="start", lsize=11.5)

    # tablet mock-up with the real museum thumbnail
    tx, ty, tw, th = 768, cy + 80, 180, 118
    s.rect(tx, ty, tw, th, "#2B3038", "#15181D", rx=12, shadow=True)
    s.image(tx + 10, ty + 10, tw - 20, th - 20, os.path.join(OUT, "..", "img", "events",
                                                             "museum.jpg"), rx=4)
    s.edge([panel.b(), (panel.cx, ty)], color=BLUE_FB, width=1.8)
    s.text(tx + tw / 2, ty + th + 20, "Pepper's tablet during Recommendation", size=11.5,
           fill=MUTED, italic=True)
    s.text(tx + tw / 2, ty + th + 36, "made transparent again at Farewell", size=11.5,
           fill=MUTED, italic=True)
    s.save(out("tablet-display"))


# =========================================================================== entry point


def main(names=None):
    for fn in FIGURES:
        if names and fn.__name__ not in names and fn.__name__.replace("_", "-") not in names:
            continue
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main(sys.argv[1:])
