"""Event image display for Pepper's tablet area.

qiBullet exposes Pepper's ``Tablet_frame`` link but not a high-level tablet API.
This helper attaches a thin textured panel to that link and swaps the texture to
the image for the recommended event. If the simulator path is unavailable, it
falls back to an OpenCV window so demos still show the selected image.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Optional

log = logging.getLogger("wp4.display")


EVENT_IMAGE_FILES = {
    "Museum": "mueseum.jpg",  # kept to match the existing filename
    "Concert": "concert.jpg",
    "Sports": "sports.jpg",
    "Food": "food.avif",
    "Outdoor": "outdoor.jpg",
    "Nightlife": "nightlife.jpg",
    "Workshop": "workshop.avif",
    "Networking": "networking.jpg",
}


class EventImageDisplay:
    """Display the top recommended event image on Pepper's tablet link."""

    def __init__(self, pepper=None, image_dir: Optional[str] = None,
                 window_fallback: bool = True) -> None:
        self.pepper = pepper
        self.image_dir = image_dir or self._default_image_dir()
        self.window_fallback = window_fallback
        self._panel_body = None
        self._converted_paths = []
        self._follow_thread = None
        self._stop_follow = threading.Event()

    @staticmethod
    def _default_image_dir() -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "..", "imgs"))

    def path_for_event(self, event: str) -> Optional[str]:
        filename = EVENT_IMAGE_FILES.get(event)
        if not filename:
            return None
        path = os.path.join(self.image_dir, filename)
        return path if os.path.exists(path) else None

    def show_event(self, event: str) -> Optional[str]:
        """Show the event image and return the path that was selected."""
        path = self.path_for_event(event)
        if not path:
            log.warning("No event image found for %r in %s.", event, self.image_dir)
            return None

        shown = self._show_on_pepper_tablet(path)
        if not shown and self.window_fallback:
            self._show_in_window(path, event)
        return path

    def clear(self) -> None:
        """Hide the tablet overlay while keeping it ready for the next event."""
        if self.pepper is not None and self._panel_body is not None:
            try:
                import pybullet as p
                p.changeVisualShape(
                    self._panel_body,
                    -1,
                    rgbaColor=[1, 1, 1, 0],
                    physicsClientId=self.pepper.physics_client,
                )
            except Exception:
                pass
        try:
            import cv2
            cv2.destroyWindow("Pepper recommendation")
            cv2.waitKey(1)
        except Exception:
            pass

    def close(self) -> None:
        self._stop_follow.set()
        if self._follow_thread is not None:
            self._follow_thread.join(timeout=1.0)
        for path in self._converted_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        self._converted_paths = []
        try:
            import cv2
            cv2.destroyWindow("Pepper recommendation")
            cv2.waitKey(1)
        except Exception:
            pass

    # -- qiBullet / PyBullet display ---------------------------------------

    def _show_on_pepper_tablet(self, path: str) -> bool:
        if self.pepper is None:
            return False
        try:
            import pybullet as p
        except Exception:
            return False

        texture_path = self._texture_compatible_path(path)
        if not texture_path:
            return False

        try:
            self._ensure_panel(p)
            texture_id = p.loadTexture(
                texture_path,
                physicsClientId=self.pepper.physics_client,
            )
            p.changeVisualShape(
                self._panel_body,
                -1,
                textureUniqueId=texture_id,
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.pepper.physics_client,
            )
            return True
        except Exception as exc:
            log.warning("Could not display event image on Pepper tablet (%s).", exc)
            return False

    def _ensure_panel(self, p) -> None:
        if self._panel_body is not None:
            return

        client = self.pepper.physics_client
        mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tablet_panel.obj")
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[1.0, 0.112, 0.063],
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=client,
        )
        self._panel_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0],
            physicsClientId=client,
        )

        tablet_link = self.pepper.link_dict.get("Tablet_frame")
        if tablet_link is None:
            raise RuntimeError("Pepper model has no Tablet_frame link.")

        self._start_following_tablet(p, tablet_link.getIndex())

    def _start_following_tablet(self, p, tablet_link_index: int) -> None:
        if self._follow_thread is not None:
            return

        def _loop():
            client = self.pepper.physics_client
            while not self._stop_follow.is_set():
                try:
                    state = p.getLinkState(
                        self.pepper.robot_model,
                        tablet_link_index,
                        computeForwardKinematics=True,
                        physicsClientId=client,
                    )
                    pos, orn = state[4], state[5]
                    mat = p.getMatrixFromQuaternion(orn)
                    # Tablet_frame's local +X points out from the screen on this
                    # Pepper URDF. Offset a little so the image sits in front of
                    # the built-in black tablet instead of being hidden by it.
                    normal = [mat[0], mat[3], mat[6]]
                    panel_pos = [
                        pos[0] + normal[0] * 0.018,
                        pos[1] + normal[1] * 0.018,
                        pos[2] + normal[2] * 0.018,
                    ]
                    p.resetBasePositionAndOrientation(
                        self._panel_body,
                        panel_pos,
                        orn,
                        physicsClientId=client,
                    )
                except Exception:
                    return
                time.sleep(0.03)

        self._follow_thread = threading.Thread(target=_loop, name="tablet-display", daemon=True)
        self._follow_thread.start()

    def _texture_compatible_path(self, path: str) -> Optional[str]:
        converted = self._make_tablet_texture(path)
        if converted:
            self._converted_paths.append(converted)
            return converted
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png"):
            return path
        return None

    @staticmethod
    def _make_tablet_texture(path: str) -> Optional[str]:
        """Create a sharp 16:9 PNG texture for PyBullet's small tablet panel."""
        try:
            from PIL import Image, ImageFilter
        except Exception:
            return EventImageDisplay._convert_with_cv2(path, suffix=".png")

        target_w, target_h = 2048, 1152
        target_ratio = float(target_w) / target_h
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                src_w, src_h = img.size
                src_ratio = float(src_w) / src_h
                if src_ratio > target_ratio:
                    new_w = int(src_h * target_ratio)
                    left = (src_w - new_w) // 2
                    img = img.crop((left, 0, left + new_w, src_h))
                elif src_ratio < target_ratio:
                    new_h = int(src_w / target_ratio)
                    top = (src_h - new_h) // 2
                    img = img.crop((0, top, src_w, top + new_h))
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
                img.save(tmp.name, format="PNG", optimize=False)
            return tmp.name
        except Exception:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
            return EventImageDisplay._convert_with_cv2(path, suffix=".png")

    # -- Fallback window ----------------------------------------------------

    def _show_in_window(self, path: str, event: str) -> None:
        try:
            import cv2
        except Exception:
            return
        img = cv2.imread(path)
        if img is None:
            converted = self._convert_with_cv2(path, suffix=".png")
            img = cv2.imread(converted) if converted else None
        if img is None:
            return
        try:
            cv2.imshow("Pepper recommendation", img)
            cv2.setWindowTitle("Pepper recommendation", event)
            cv2.waitKey(1)
        except Exception:
            pass

    @staticmethod
    def _convert_with_cv2(path: str, suffix: str = ".png") -> Optional[str]:
        pil_path = EventImageDisplay._convert_with_pillow(path, suffix)
        if pil_path:
            return pil_path

        try:
            import cv2
        except Exception:
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        ok = cv2.imwrite(tmp.name, img)
        if not ok:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
            return None
        return tmp.name

    @staticmethod
    def _convert_with_pillow(path: str, suffix: str = ".png") -> Optional[str]:
        try:
            from PIL import Image
        except Exception:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        try:
            with Image.open(path) as img:
                img.convert("RGBA").save(tmp.name)
            return tmp.name
        except Exception:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
            return None
