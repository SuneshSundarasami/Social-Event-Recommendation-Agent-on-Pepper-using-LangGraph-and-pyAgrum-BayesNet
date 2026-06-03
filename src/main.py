"""WP1 orchestrator — runs the perception + state machine spine.

The dialogue (WP2), recommender (WP3) and behaviour (WP4) packages are wired in
as stubs for now; swap them for the real implementations as they land.

Examples:
    # Headless demo, no camera / no qiBullet — one full interaction cycle:
    python main.py --source scripted

    # Local webcam face detection, interactive console dialogue, loop forever:
    python main.py --source webcam --interactive

    # Pepper's simulated top camera in qiBullet:
    python main.py --source pepper
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Allow `python main.py` from anywhere by putting this dir (src/) on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_machine import InteractionFSM  # noqa: E402
from stubs import ConsoleBehaviour, StubDialogueManager, StubRecommender  # noqa: E402


def build_perception(args):
    """Construct a Perception implementation from CLI args.

    Returns (perception, cleanup_callable).
    """
    if args.source == "scripted":
        from perception.face_detector import ScriptedPerception

        # Become stable quickly; "leave" shortly after the recommendation so the
        # acknowledgement wait returns promptly and the demo ends cleanly.
        return ScriptedPerception(stable_after=3, leaves_after=12), (lambda: None)

    if args.source == "webcam":
        from perception.face_detector import HaarFaceDetector, WebcamSource

        source = WebcamSource(args.camera_index)
        det = HaarFaceDetector(source)
        return det, det.release

    if args.source == "pepper":
        from qibullet import SimulationManager
        from perception.face_detector import HaarFaceDetector, PepperCameraSource

        sim = SimulationManager()
        client = sim.launchSimulation(gui=not args.headless)
        pepper = sim.spawnPepper(client, spawn_ground_plane=True)
        source = PepperCameraSource(pepper)
        det = HaarFaceDetector(source)

        def cleanup():
            det.release()
            sim.stopSimulation(client)

        return det, cleanup

    raise ValueError("Unknown source: %s" % args.source)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="WP1: perception + interaction FSM.")
    parser.add_argument(
        "--source",
        choices=["scripted", "webcam", "pepper"],
        default="scripted",
        help="Perception source (default: scripted, needs no hardware).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--headless", action="store_true", help="qiBullet without GUI.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask preference questions on the console instead of using defaults.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N interactions (default: 1 for scripted, unlimited otherwise).",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    perception, cleanup = build_perception(args)

    dialogue = StubDialogueManager(interactive=args.interactive)
    recommender = StubRecommender()
    behaviour = ConsoleBehaviour()

    # Scripted mode is a quick self-contained demo: fast polling, short ack wait.
    if args.source == "scripted":
        fsm = InteractionFSM(
            perception, dialogue, recommender, behaviour,
            idle_poll=0.05, ack_timeout=2.0,
        )
        max_cycles = args.max_cycles if args.max_cycles is not None else 1
    else:
        fsm = InteractionFSM(perception, dialogue, recommender, behaviour)
        max_cycles = args.max_cycles

    print("=== Social Event Recommendation Agent - WP1 (source=%s) ===" % args.source)
    try:
        fsm.run(max_cycles=max_cycles)
    finally:
        cleanup()
    print("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
