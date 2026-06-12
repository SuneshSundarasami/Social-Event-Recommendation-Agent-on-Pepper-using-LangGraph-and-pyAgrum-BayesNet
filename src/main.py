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


def _spawn_pepper(args):
    """Launch qiBullet, spawn Pepper, and frame the GUI camera on its face."""
    import pybullet as p
    from qibullet import SimulationManager

    sim = SimulationManager()
    client = sim.launchSimulation(gui=not args.headless)
    pepper = sim.spawnPepper(client, spawn_ground_plane=True)
    pepper.goToPosture("Stand", 0.6)

    if not args.headless:
        # Set camera to face Pepper's face from the front.
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=0,
            cameraTargetPosition=[0, 0, 1.4],
            physicsClientId=client,
        )
    return sim, client, pepper


def build_components(args):
    """Construct (perception, behaviour, cleanup) from CLI args."""
    if args.source == "scripted":
        from perception.face_detector import ScriptedPerception

        # Become stable quickly; "leave" shortly after the recommendation so the
        # acknowledgement wait returns promptly and the demo ends cleanly.
        perception = ScriptedPerception(stable_after=3, leaves_after=12)
        return perception, ConsoleBehaviour(), (lambda: None)

    if args.source == "webcam":
        from perception.face_detector import HaarFaceDetector, ThreadedPerception, WebcamSource

        det = HaarFaceDetector(WebcamSource(args.camera_index))
        # Threaded preview so the window stays live during the (blocking) dialogue.
        perception = ThreadedPerception(det, preview=True) if args.preview else det
        return perception, ConsoleBehaviour(), perception.release

    if args.source in ("pepper", "hybrid"):
        from behaviour.pepper_behaviour import PepperBehaviour
        from perception.face_detector import HaarFaceDetector

        sim, client, pepper = _spawn_pepper(args)
        behaviour = PepperBehaviour(pepper)

        if args.source == "pepper":
            from perception.face_detector import PepperCameraSource

            perception = HaarFaceDetector(PepperCameraSource(pepper))
        else:  # hybrid: Pepper's body in sim, face detection from the webcam
            from perception.face_detector import ThreadedPerception, WebcamSource

            det = HaarFaceDetector(WebcamSource(args.camera_index))
            perception = ThreadedPerception(det, preview=True)

        def cleanup():
            behaviour.shutdown()
            perception.release()
            sim.stopSimulation(client)

        return perception, behaviour, cleanup

    raise ValueError("Unknown source: %s" % args.source)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="WP1: perception + interaction FSM.")
    parser.add_argument(
        "--source",
        choices=["scripted", "webcam", "pepper", "hybrid"],
        default="scripted",
        help="Perception source: scripted (no hardware), webcam, pepper "
             "(qiBullet camera), or hybrid (Pepper in qiBullet + webcam detection).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a webcam preview window with detection boxes (auto-on for hybrid).",
    )
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

    perception, behaviour, cleanup = build_components(args)

    dialogue = StubDialogueManager(interactive=args.interactive)
    # WP3: use the real Bayesian recommender; fall back to the stub if pyAgrum
    # or the network is unavailable.
    try:
        from recommender.bayesian_network import BayesianRecommender

        recommender = BayesianRecommender()
    except Exception as exc:
        logging.warning("Bayesian recommender unavailable (%s); using stub.", exc)
        recommender = StubRecommender()

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
