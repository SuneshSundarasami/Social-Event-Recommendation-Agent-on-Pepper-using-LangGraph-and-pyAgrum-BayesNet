"""Face verification of authorized users (WP1, optional requirement #1).

Uses OpenCV's built-in LBPH (Local Binary Patterns Histograms) recognizer --
classic, CPU-only, no deep-learning dependency, and trains from a handful of
photos per person. Enrolled faces live under ``known_faces/<name>/*.jpg``
(create them with ``enroll_face.py``).

Fails soft: with nobody enrolled (or ``cv2.face`` unavailable) ``available``
is False and ``identify()`` always returns ``None``, so the FSM proceeds
exactly as it does today for an unrecognized/guest face.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

log = logging.getLogger("wp1.face_recognizer")

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore

FACE_SIZE = (200, 200)  # all enrollment/prediction crops are normalised to this


def _default_known_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "known_faces")


def preprocess(gray_frame, box) -> "cv2.typing.MatLike":
    """Crop a face box out of a grayscale frame and normalise it for LBPH."""
    x, y, w, h = box
    crop = gray_frame[y:y + h, x:x + w]
    return cv2.resize(crop, FACE_SIZE)


class FaceRecognizer:
    """Trains on ``known_dir`` at construction time; ``identify()`` per frame."""

    def __init__(self, known_dir: Optional[str] = None,
                 confidence_threshold: float = 75.0) -> None:
        self.known_dir = known_dir or _default_known_dir()
        self.confidence_threshold = confidence_threshold
        self.available = False
        self._recognizer = None
        self._label_to_name: List[str] = []

        if cv2 is None or not hasattr(cv2, "face"):
            log.warning("cv2.face unavailable (need opencv-contrib-python); "
                       "face verification disabled.")
            return
        self._train()

    def _enrolled_people(self) -> List[str]:
        if not os.path.isdir(self.known_dir):
            return []
        return sorted(
            name for name in os.listdir(self.known_dir)
            if os.path.isdir(os.path.join(self.known_dir, name))
        )

    def _train(self) -> None:
        people = self._enrolled_people()
        faces, labels = [], []
        for label, name in enumerate(people):
            person_dir = os.path.join(self.known_dir, name)
            for fname in sorted(os.listdir(person_dir)):
                path = os.path.join(person_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(cv2.resize(img, FACE_SIZE))
                labels.append(label)

        if not faces:
            log.info("No enrolled faces under %s; face verification disabled "
                     "until someone runs enroll_face.py.", self.known_dir)
            return

        import numpy as np

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        self._recognizer = recognizer
        self._label_to_name = people
        self.available = True
        log.info("Face recognizer trained on %d image(s) for %d authorized "
                 "user(s): %s", len(faces), len(people), ", ".join(people))

    def identify(self, gray_frame, box: Tuple[int, int, int, int]) -> Optional[str]:
        """Return the authorized user's name for this face box, or None."""
        if not self.available:
            return None
        face = preprocess(gray_frame, box)
        label, confidence = self._recognizer.predict(face)
        # LBPH confidence is a distance: LOWER means a closer match.
        if confidence <= self.confidence_threshold:
            return self._label_to_name[label]
        return None

    def enroll_sample(self, frame, box: Tuple[int, int, int, int], name: str) -> str:
        """Save one live BGR frame's face crop under known_faces/<name>/.

        Used for consent-based, in-conversation enrollment of a new,
        previously-unrecognized person (as opposed to the offline
        ``enroll_face.py`` CLI). Returns the saved file path.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        face = preprocess(gray, box)
        person_dir = os.path.join(self.known_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        index = len(os.listdir(person_dir))
        path = os.path.join(person_dir, "%03d.jpg" % index)
        cv2.imwrite(path, face)
        return path

    def retrain(self) -> None:
        """Re-scan known_dir and rebuild the model (e.g. after enrolling someone
        new), so recognition works for the rest of the current session too."""
        self._train()
