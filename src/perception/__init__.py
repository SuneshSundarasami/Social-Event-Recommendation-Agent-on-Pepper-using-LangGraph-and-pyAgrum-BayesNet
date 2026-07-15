"""WP1 perception package: camera sources + face detection with smoothing."""

from perception.face_detector import (  # noqa: F401
    HaarFaceDetector,
    PepperCameraSource,
    PreviewPerception,
    ScriptedPerception,
    WebcamSource,
)
from perception.face_recognizer import FaceRecognizer  # noqa: F401
