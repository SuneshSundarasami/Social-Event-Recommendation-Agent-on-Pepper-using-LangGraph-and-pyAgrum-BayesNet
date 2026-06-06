"""Standalone perception diagnostic (WP1).

Runs the real Haar face detector against a webcam or Pepper's qiBullet camera so
you can confirm detection works *before* wiring it into the FSM. Shows a live
window with detection boxes and the smoothing-window status:

    detected  -- a face was found in THIS frame
    present   -- enough recent frames had a face (interaction stays alive)
    STABLE    -- window full + >= threshold frames -> would trigger Greeting

Press 'q' (or Esc) in the window to quit. With --no-window it prints a status
line per frame instead (useful over SSH / headless).

Examples:
    python check_perception.py --source webcam
    python check_perception.py --source webcam --no-window
    python check_perception.py --source pepper
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
except Exception:
    cv2 = None

from perception.face_detector import HaarFaceDetector, PepperCameraSource, WebcamSource


def build_source(args):
    if args.source == "webcam":
        return WebcamSource(args.camera_index), (lambda: None)
    if args.source == "pepper":
        from qibullet import SimulationManager

        sim = SimulationManager()
        client = sim.launchSimulation(gui=not args.headless)
        pepper = sim.spawnPepper(client, spawn_ground_plane=True)
        return PepperCameraSource(pepper), (lambda: sim.stopSimulation(client))
    raise ValueError("Unknown source: %s" % args.source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WP1 perception diagnostic.")
    parser.add_argument("--source", choices=["webcam", "pepper"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="qiBullet without GUI.")
    parser.add_argument("--no-window", action="store_true", help="Print status, no GUI window.")
    args = parser.parse_args(argv)

    if cv2 is None:
        print("ERROR: OpenCV (cv2) is not installed. Run `uv sync` in Project/ first.")
        return 1

    source, cleanup = build_source(args)
    det = HaarFaceDetector(source)

    print("Perception diagnostic running (source=%s). Ctrl-C / 'q' to quit." % args.source)
    show = not args.no_window
    try:
        while True:
            detected = det.tick()
            present = det.is_present()
            stable = det.face_stable()

            if show and det.last_frame is not None:
                frame = det.last_frame.copy()
                for (x, y, w, h) in det.last_faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = "detected=%s present=%s %s" % (
                    detected, present, "STABLE" if stable else "")
                color = (0, 255, 0) if stable else (0, 200, 255) if present else (0, 0, 255)
                cv2.putText(frame, status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("WP1 perception check", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            else:
                print("detected=%-5s present=%-5s stable=%-5s faces=%d" % (
                    detected, present, stable, len(det.last_faces)))
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        det.release()
        cleanup()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
