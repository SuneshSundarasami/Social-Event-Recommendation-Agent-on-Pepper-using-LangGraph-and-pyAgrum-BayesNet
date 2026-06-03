"""Face detection with a smoothing window (WP1).

Provides:
  * Frame sources  -- WebcamSource, PepperCameraSource (qiBullet top camera).
  * HaarFaceDetector -- OpenCV Haar-cascade detector implementing the
    ``Perception`` contract, with a five-frame smoothing window so brief missed
    detections do not cause the interaction to flicker on/off.
  * ScriptedPerception -- a camera-free stand-in for testing the state machine
    (and demoing the flow) without qiBullet or a webcam.

OpenCV is imported lazily so ScriptedPerception works in environments where
``cv2`` is not installed.
"""

from __future__ import annotations

import collections
import logging
from typing import Deque, Optional

log = logging.getLogger("wp1.perception")

# cv2 is optional: only the real camera detectors need it.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class WebcamSource:
    """A local webcam frame source (cv2.VideoCapture)."""

    def __init__(self, index: int = 0) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for WebcamSource.")
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam index %d." % index)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class PepperCameraSource:
    """Frame source backed by Pepper's simulated camera in qiBullet.

    Pass the spawned ``pepper`` object; the source subscribes to the requested
    camera and returns BGR frames compatible with OpenCV.
    """

    def __init__(self, pepper, camera_id: Optional[int] = None, fps: float = 15.0) -> None:
        # Imported here so the module loads without qibullet present.
        from qibullet import Camera, PepperVirtual

        self.pepper = pepper
        if camera_id is None:
            camera_id = PepperVirtual.ID_CAMERA_TOP
        self.handle = pepper.subscribeCamera(
            camera_id, resolution=Camera.K_QVGA, fps=fps
        )

    def read(self):
        return self.pepper.getCameraFrame(self.handle)

    def release(self) -> None:
        try:
            self.pepper.unsubscribeCamera(self.handle)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Haar-cascade detector with smoothing window
# ---------------------------------------------------------------------------

class HaarFaceDetector:
    """Detects faces in frames from a source, smoothing over a sliding window.

    ``face_stable()`` (Idle -> Greeting trigger) requires the window to be full
    and at least ``stable_threshold`` of those frames to contain a face.
    ``is_present()`` uses a looser bar so brief misses during the interaction do
    not look like the user left.
    """

    def __init__(
        self,
        source,
        window: int = 5,
        stable_threshold: int = 4,
        present_threshold: int = 2,
        cascade_path: Optional[str] = None,
        min_face_size: int = 60,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for HaarFaceDetector.")
        self.source = source
        self.stable_threshold = stable_threshold
        self.present_threshold = present_threshold
        self.min_face_size = min_face_size
        self._window: Deque[bool] = collections.deque(maxlen=window)

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade from %r." % cascade_path)

    def tick(self) -> bool:
        frame = self.source.read()
        detected = False
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            detected = len(faces) > 0
        self._window.append(detected)
        return detected

    def face_stable(self) -> bool:
        full = len(self._window) == self._window.maxlen
        return full and self._window.count(True) >= self.stable_threshold

    def is_present(self) -> bool:
        return self._window.count(True) >= self.present_threshold

    def reset(self) -> None:
        self._window.clear()

    def release(self) -> None:
        self.source.release()


# ---------------------------------------------------------------------------
# Scripted (camera-free) perception for tests / headless demo
# ---------------------------------------------------------------------------

class ScriptedPerception:
    """Deterministic ``Perception`` stand-in driven by tick count.

    A face becomes detected from tick ``stable_after`` onward, and (optionally)
    the user "leaves" from tick ``leaves_after`` onward. Useful for exercising
    the full FSM cycle without any camera.
    """

    def __init__(self, stable_after: int = 3, leaves_after: Optional[int] = None) -> None:
        self.stable_after = stable_after
        self.leaves_after = leaves_after
        self._t = 0
        self._present = False

    def tick(self) -> bool:
        self._t += 1
        present = self._t >= self.stable_after
        if self.leaves_after is not None and self._t >= self.leaves_after:
            present = False
        self._present = present
        return present

    def face_stable(self) -> bool:
        return self._present

    def is_present(self) -> bool:
        return self._present

    def reset(self) -> None:
        # Keep the tick counter so a scripted "leave" stays consistent across
        # an Idle reset; only the presence latch is cleared.
        self._present = False

    def release(self) -> None:
        pass
