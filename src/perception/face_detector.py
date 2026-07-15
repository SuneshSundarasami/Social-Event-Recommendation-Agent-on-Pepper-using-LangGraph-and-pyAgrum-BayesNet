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
import os
import threading
import time
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
        # On Windows the default MSMF backend often opens the device but then
        # can't grab frames (CvCapture_MSMF::grabFrame ... -1072875772). Prefer
        # DirectShow there, which is far more reliable, and fall back to the
        # platform default if DirectShow can't deliver a frame.
        self.cap = self._open(index)
        if self.cap is None:
            raise RuntimeError("Could not open webcam index %d." % index)

    @staticmethod
    def _open(index):
        backends = []
        if os.name == "nt":
            backends.append(getattr(cv2, "CAP_DSHOW", 0))
        backends.append(getattr(cv2, "CAP_ANY", 0))
        for backend in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, _ = cap.read()  # MSMF can report isOpened() yet fail to stream
                if ok:
                    return cap
            cap.release()
        return None

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
        recognizer=None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for HaarFaceDetector.")
        self.source = source
        self.stable_threshold = stable_threshold
        self.present_threshold = present_threshold
        self.min_face_size = min_face_size
        self.recognizer = recognizer  # optional FaceRecognizer (WP1 requirement #1)
        self._window: Deque[bool] = collections.deque(maxlen=window)

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade from %r." % cascade_path)

        # Last frame/faces from tick() — handy for visual diagnostics.
        self.last_frame = None
        self.last_faces = []
        self.last_identity: Optional[str] = None

    def tick(self) -> bool:
        frame = self.source.read()
        detected = False
        faces = []
        identity = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            detected = len(faces) > 0
            if detected and self.recognizer is not None:
                largest = max(faces, key=lambda b: b[2] * b[3])
                identity = self.recognizer.identify(gray, largest)
        self.last_frame = frame
        self.last_faces = faces
        self.last_identity = identity
        self._window.append(detected)
        return detected

    def face_stable(self) -> bool:
        full = len(self._window) == self._window.maxlen
        return full and self._window.count(True) >= self.stable_threshold

    def is_present(self) -> bool:
        return self._window.count(True) >= self.present_threshold

    def identify(self) -> Optional[str]:
        """Authorized user's name for the current face, or None (WP1 requirement #1)."""
        return self.last_identity

    def reset(self) -> None:
        self._window.clear()

    def release(self) -> None:
        self.source.release()


# ---------------------------------------------------------------------------
# Preview wrapper — shows a live OpenCV window with detection overlay
# ---------------------------------------------------------------------------

class PreviewPerception:
    """Wraps any detector exposing ``last_frame``/``last_faces`` (e.g.
    ``HaarFaceDetector``) and renders a live preview window on every tick.

    Transparent decorator: forwards the full ``Perception`` interface, so it can
    be dropped in anywhere a detector is expected. The window updates whenever
    the FSM ticks perception (idle loop, presence checks, ack wait).
    """

    def __init__(self, inner, window_name: str = "Webcam - face detection") -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for PreviewPerception.")
        self.inner = inner
        self.window_name = window_name

    def tick(self) -> bool:
        detected = self.inner.tick()
        frame = getattr(self.inner, "last_frame", None)
        if frame is not None:
            disp = frame.copy()
            for (x, y, w, h) in getattr(self.inner, "last_faces", []):
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            stable = self.inner.face_stable()
            present = self.inner.is_present()
            if stable:
                label, color = "STABLE", (0, 255, 0)
            elif present:
                label, color = "present", (0, 200, 255)
            else:
                label, color = "searching", (0, 0, 255)
            cv2.putText(disp, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(self.window_name, disp)
            cv2.waitKey(1)
        return detected

    def face_stable(self) -> bool:
        return self.inner.face_stable()

    def is_present(self) -> bool:
        return self.inner.is_present()

    def identify(self):
        return getattr(self.inner, "identify", lambda: None)()

    @property
    def last_frame(self):
        return getattr(self.inner, "last_frame", None)

    @property
    def last_faces(self):
        return getattr(self.inner, "last_faces", [])

    @property
    def recognizer(self):
        return getattr(self.inner, "recognizer", None)

    def reset(self) -> None:
        self.inner.reset()

    def release(self) -> None:
        self.inner.release()
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Threaded wrapper — keeps capture + preview alive on a background thread
# ---------------------------------------------------------------------------

class ThreadedPerception:
    """Runs an inner detector on its own thread so detection and the preview
    window keep updating even when the main thread is blocked (e.g. waiting on
    console input during the dialogue, or sleeping through a gesture).

    Without this, the OpenCV window only refreshes when the FSM happens to call
    ``tick()`` and so appears to "freeze" the moment the interaction starts.

    All OpenCV highgui calls happen on this one background thread (required for
    stability); the FSM-facing methods are cheap, lock-guarded accessors.
    """

    def __init__(self, inner, preview: bool = True, fps: float = 15.0,
                 window_name: str = "Webcam - face detection") -> None:
        self.inner = inner
        self.preview = preview and cv2 is not None
        self.window_name = window_name
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._detected = False
        self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                detected = self.inner.tick()
                self._detected = detected
                frame = getattr(self.inner, "last_frame", None)
                faces = list(getattr(self.inner, "last_faces", []))
                stable = self.inner.face_stable()
                present = self.inner.is_present()
            if self.preview and frame is not None:
                disp = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if stable:
                    label, color = "STABLE", (0, 255, 0)
                elif present:
                    label, color = "present", (0, 200, 255)
                else:
                    label, color = "searching", (0, 0, 255)
                cv2.putText(disp, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow(self.window_name, disp)
                cv2.waitKey(1)
            if self._period:
                time.sleep(self._period)
        if self.preview:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except Exception:
                pass

    # The background thread does the real work; tick() just reports latest state.
    def tick(self) -> bool:
        return self._detected

    def face_stable(self) -> bool:
        with self._lock:
            return self.inner.face_stable()

    def is_present(self) -> bool:
        with self._lock:
            return self.inner.is_present()

    def identify(self):
        with self._lock:
            return getattr(self.inner, "identify", lambda: None)()

    @property
    def last_frame(self):
        with self._lock:
            return getattr(self.inner, "last_frame", None)

    @property
    def last_faces(self):
        with self._lock:
            return getattr(self.inner, "last_faces", [])

    @property
    def recognizer(self):
        with self._lock:
            return getattr(self.inner, "recognizer", None)

    def reset(self) -> None:
        with self._lock:
            self.inner.reset()

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.inner.release()


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
