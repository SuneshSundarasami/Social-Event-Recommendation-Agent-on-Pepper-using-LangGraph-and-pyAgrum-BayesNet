"""Enroll an authorized user's face for verification (WP1, optional #1).

Opens the webcam, detects a face each frame with the same Haar cascade the
robot uses at runtime, and saves cropped/normalised grayscale samples to
``perception/known_faces/<name>/`` for ``FaceRecognizer`` to train on.

Example:
    python enroll_face.py --name Sunesh
    python enroll_face.py --name Sunesh --count 30 --camera-index 1
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

from perception.face_detector import HaarFaceDetector, WebcamSource
from perception.face_recognizer import FACE_SIZE, preprocess


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Enroll a face for WP1 verification.")
    parser.add_argument("--name", required=True, help="Authorized user's name.")
    parser.add_argument("--count", type=int, default=20, help="Samples to capture.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.3,
                        help="Seconds between captured samples.")
    args = parser.parse_args(argv)

    if cv2 is None:
        print("ERROR: OpenCV (cv2) is not installed. Run `uv sync` in Project/ first.")
        return 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "perception", "known_faces", args.name)
    os.makedirs(out_dir, exist_ok=True)
    existing = len(os.listdir(out_dir))

    det = HaarFaceDetector(WebcamSource(args.camera_index))
    print("Enrolling '%s' - look at the camera. Press 'q' to stop early." % args.name)

    saved = 0
    last_capture = 0.0
    try:
        while saved < args.count:
            det.tick()
            frame = det.last_frame
            if frame is not None and det.last_faces is not None and len(det.last_faces):
                box = max(det.last_faces, key=lambda b: b[2] * b[3])  # largest face
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                now = time.time()
                if now - last_capture >= args.interval:
                    gray = cv2.cvtColor(det.last_frame, cv2.COLOR_BGR2GRAY)
                    sample = preprocess(gray, box)
                    path = os.path.join(out_dir, "%03d.jpg" % (existing + saved))
                    cv2.imwrite(path, sample)
                    saved += 1
                    last_capture = now
                    print("  captured %d/%d" % (saved, args.count))

                cv2.putText(frame, "%d/%d" % (saved, args.count), (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Enroll face - %s" % args.name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        det.release()
        cv2.destroyAllWindows()

    print("Saved %d sample(s) to %s" % (saved, out_dir))
    print("Restart main.py with a camera source to use verification "
         "(FACE_SIZE=%s)." % (FACE_SIZE,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
