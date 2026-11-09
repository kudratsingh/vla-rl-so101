"""The exact image the policy sees from the real front camera. Shared by WO-06 and WO-07.

Pipeline: real RGB frame (``cameras.front`` width x height) -> uint8 (S, S, 3) with
S = ``cameras.front.crop_to`` from the config (224, OpenVLA-OFT's input; must equal the sim's
``env.image_size`` - ``sim_mismatches`` checks):

1. Undistort with the calibrated intrinsics, keeping the same K (the sim camera is a distortion-free
   pinhole).
2. Crop the window that subtends the sim camera's field of view (``cameras.front.fov_deg``, must
   equal the sim's ``env.camera.fov_deg``), centred on the principal point:
   half-width = fx * tan(fov / 2), half-height = fy * tan(fov / 2). The sim renders a square image
   with a symmetric pinhole, so its horizontal and vertical fov are equal.
3. Resize to S x S with ``cv2.INTER_AREA``.

Without intrinsics (before ``scripts/wo06_camera_intrinsics.py`` has run) step 1 is skipped and
step 2 is a centred square crop of min(width, height). That fallback is UNCALIBRATED and logged.

Frames carry a ``time.perf_counter`` capture stamp (LeRobot's OpenCVCamera clock) for the deadman.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

import cv2
import numpy as np
import numpy.typing as npt

from src.config import Config

log = logging.getLogger(__name__)

ImageU8 = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole + OpenCV distortion (k1, k2, p1, p2, k3) for one camera at width x height."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...]
    width: int
    height: int
    rms_px: float | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Intrinsics:
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            dist=tuple(float(v) for v in d.get("dist", ())),
            width=int(d["width"]),
            height=int(d["height"]),
            rms_px=None if d.get("rms_px") is None else float(d["rms_px"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": round(self.fx, 3),
            "fy": round(self.fy, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "dist": [round(v, 6) for v in self.dist],
            "width": self.width,
            "height": self.height,
            "rms_px": None if self.rms_px is None else round(self.rms_px, 4),
        }

    def camera_matrix(self) -> npt.NDArray[np.float64]:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def fov_deg(self) -> tuple[float, float]:
        """(horizontal, vertical) full field of view of the whole image."""
        return (
            math.degrees(2 * math.atan(self.width / (2 * self.fx))),
            math.degrees(2 * math.atan(self.height / (2 * self.fy))),
        )


def fov_crop_window(
    intrinsics: Intrinsics | None, width: int, height: int, fov_deg: float
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel window matching the sim fov; centred square if uncalibrated."""
    if intrinsics is None:
        side = min(width, height)
        x0, y0 = (width - side) // 2, (height - side) // 2
        return x0, y0, x0 + side, y0 + side
    t = math.tan(math.radians(fov_deg) / 2)
    hw, hh = intrinsics.fx * t, intrinsics.fy * t
    x0, x1 = round(intrinsics.cx - hw), round(intrinsics.cx + hw)
    y0, y1 = round(intrinsics.cy - hh), round(intrinsics.cy + hh)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"the sim camera's {fov_deg:.1f} deg view needs a {2 * hw:.0f}x{2 * hh:.0f} px window "
            f"around the principal point ({intrinsics.cx:.0f}, {intrinsics.cy:.0f}), but the real "
            f"image is {width}x{height}: the real camera's field of view is too narrow or "
            "off-centre. Use a wider lens, or change fov_deg in BOTH configs/sim.yaml and "
            "configs/real.yaml (and re-render sim data)."
        )
    return x0, y0, x1, y1


def sim_mismatches(real_cfg: Config, sim_cfg: Config, camera: str = "front") -> list[str]:
    """Where the real camera config disagrees with the sim camera the policy was trained on."""
    cam = real_cfg["cameras"][camera]
    env = sim_cfg.get("env", {})
    sim_cam = env.get("camera", {})
    pairs = [
        ("crop_to vs env.image_size", cam.get("crop_to"), env.get("image_size")),
        ("fov_deg vs env.camera.fov_deg", cam.get("fov_deg"), sim_cam.get("fov_deg")),
        ("pose vs env.camera.pose", cam.get("pose"), sim_cam.get("pose")),
    ]
    return [f"{what}: real {a!r} != sim {b!r}" for what, a, b in pairs if a != b]


class PolicyImagePipeline:
    """Real frame -> policy image. ``__call__`` is the one function WO-07 must use."""

    def __init__(
        self,
        width: int,
        height: int,
        out_size: int,
        fov_deg: float,
        intrinsics: Intrinsics | None = None,
    ):
        if intrinsics is not None and (intrinsics.width, intrinsics.height) != (width, height):
            raise ValueError(
                f"intrinsics are for {intrinsics.width}x{intrinsics.height}, camera streams "
                f"{width}x{height}; recalibrate at the streaming resolution"
            )
        self.width, self.height, self.out_size = int(width), int(height), int(out_size)
        self.fov_deg = float(fov_deg)
        self.intrinsics = intrinsics
        self.window = fov_crop_window(intrinsics, self.width, self.height, self.fov_deg)
        self._maps: tuple[Any, Any] | None = None
        if intrinsics is not None and any(abs(d) > 0 for d in intrinsics.dist):
            k = intrinsics.camera_matrix()
            dist = np.asarray(intrinsics.dist, dtype=np.float64)
            m1, m2 = cv2.initUndistortRectifyMap(
                k, dist, np.eye(3), k, (self.width, self.height), cv2.CV_32FC1
            )
            self._maps = (m1, m2)
        if intrinsics is None:
            log.warning(
                "camera pipeline UNCALIBRATED: centred square crop, no undistortion "
                "(run scripts/wo06_camera_intrinsics.py)"
            )

    @property
    def calibrated(self) -> bool:
        return self.intrinsics is not None

    @classmethod
    def from_config(cls, cfg: Config, camera: str = "front") -> PolicyImagePipeline:
        cam = cfg["cameras"][camera]
        intr = cam.get("intrinsics")
        return cls(
            width=int(cam["width"]),
            height=int(cam["height"]),
            out_size=int(cam["crop_to"]),
            fov_deg=float(cam["fov_deg"]),
            intrinsics=Intrinsics.from_dict(intr) if intr else None,
        )

    def __call__(self, frame: npt.ArrayLike) -> ImageU8:
        img = np.asarray(frame)
        if img.dtype != np.uint8 or img.shape != (self.height, self.width, 3):
            raise ValueError(
                f"expected uint8 ({self.height}, {self.width}, 3) RGB, got {img.dtype} {img.shape}"
            )
        if self._maps is not None:
            img = np.asarray(cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR))
        x0, y0, x1, y1 = self.window
        crop = img[y0:y1, x0:x1]
        out = cv2.resize(crop, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(out, dtype=np.uint8)


# ------------------------------------------------------------------------------------------------
# Frame source
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    image: ImageU8  # (H, W, 3) RGB
    capture_time: float  # time.perf_counter() when the frame was grabbed


class CameraFault(RuntimeError):
    """No fresh frame. A control loop treats this as a lost frame (deadman)."""


class OpenCVFrameSource:
    """``cameras.<name>`` through LeRobot 0.4.4's OpenCVCamera (background read thread, RGB)."""

    def __init__(self, cam_cfg: Mapping[str, Any], timeout_ms: float = 200.0):
        self.cam_cfg = dict(cam_cfg)
        self.timeout_ms = float(timeout_ms)
        self._cam: Any = None

    def connect(self) -> None:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        cfg = OpenCVCameraConfig(
            index_or_path=self.cam_cfg["index"],
            fps=int(self.cam_cfg["fps"]),
            width=int(self.cam_cfg["width"]),
            height=int(self.cam_cfg["height"]),
        )
        self._cam = OpenCVCamera(cfg)
        self._cam.connect()

    def read(self) -> Frame:
        if self._cam is None:
            raise CameraFault("camera not connected")
        try:
            self._cam.async_read(timeout_ms=self.timeout_ms)  # waits for a fresh frame
            with self._cam.frame_lock:  # image and its stamp as one consistent pair
                image, stamp = self._cam.latest_frame, self._cam.latest_timestamp
        except Exception as e:
            raise CameraFault(f"camera read failed: {type(e).__name__}: {e}") from e
        if image is None or stamp is None:
            raise CameraFault("camera returned no frame")
        return Frame(np.asarray(image, dtype=np.uint8), float(stamp))

    def disconnect(self) -> None:
        if self._cam is not None:
            try:
                self._cam.disconnect()
            finally:
                self._cam = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()


class FrameReader(Protocol):
    """One camera: ``OpenCVFrameSource`` or a test fake."""

    def connect(self) -> None: ...

    def read(self) -> Frame: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class RigFrame:
    """One frame from every camera of a ``CameraRig``, read on the same control tick."""

    images: dict[str, ImageU8]  # camera name -> (H, W, 3) RGB
    capture_times: dict[str, float]  # camera name -> time.perf_counter() stamp

    @property
    def capture_time(self) -> float:
        """The OLDEST camera's stamp: the deadman's frame-age check covers every camera."""
        return min(self.capture_times.values())


class CameraRig:
    """Named cameras read together (``src/robot/convention.py``: front + wrist).

    ``read`` returns a frame only when EVERY camera delivered one; a fault on any camera raises
    ``CameraFault`` naming it, so a control loop treats it as a lost frame (deadman) and nothing
    is recorded or acted on with one camera missing.
    """

    def __init__(self, sources: Mapping[str, FrameReader]):
        if not sources:
            raise ValueError("a camera rig needs at least one camera")
        self.sources = dict(sources)

    @classmethod
    def from_config(cls, cfg: Config, names: Sequence[str]) -> CameraRig:
        cams = cfg.get("cameras") or {}
        missing = [n for n in names if not cams.get(n) or cams[n].get("enabled", True) is False]
        if missing:
            raise ValueError(f"cameras {missing} missing or disabled in the config")
        return cls({n: OpenCVFrameSource(cams[n]) for n in names})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.sources)

    def connect(self) -> None:
        opened: list[FrameReader] = []
        try:
            for src in self.sources.values():
                src.connect()
                opened.append(src)
        except BaseException:
            for src in opened:
                src.disconnect()
            raise

    def read(self) -> RigFrame:
        images: dict[str, ImageU8] = {}
        stamps: dict[str, float] = {}
        for name, src in self.sources.items():
            try:
                f = src.read()
            except CameraFault as e:
                raise CameraFault(f"{name} camera: {e}") from e
            images[name], stamps[name] = f.image, float(f.capture_time)
        return RigFrame(images, stamps)

    def disconnect(self) -> None:
        errors: list[BaseException] = []
        for src in self.sources.values():
            try:
                src.disconnect()
            except Exception as e:  # noqa: BLE001 - close every camera, then report
                errors.append(e)
        if errors:
            raise errors[0]

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()


# ------------------------------------------------------------------------------------------------
# Checkerboard intrinsics
# ------------------------------------------------------------------------------------------------
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def find_checkerboard(image_rgb: npt.ArrayLike, pattern: tuple[int, int]) -> Any:
    """Sub-pixel inner-corner positions (N, 1, 2) float32, or None if not found.

    ``pattern`` = inner corners (columns, rows), e.g. (9, 6) for a 10 x 7-square board.
    """
    img = np.asarray(image_rgb, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX)


def calibrate_intrinsics(
    images: Sequence[npt.ArrayLike], pattern: tuple[int, int], square_m: float, min_views: int = 5
) -> tuple[Intrinsics, list[bool]]:
    """OpenCV checkerboard calibration. Returns (intrinsics, per-image detected flags)."""
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    obj_pts: list[Any] = []
    img_pts: list[Any] = []
    found: list[bool] = []
    size: tuple[int, int] | None = None
    for im in images:
        arr = np.asarray(im)
        hw = (arr.shape[1], arr.shape[0])
        if size is None:
            size = hw
        elif hw != size:
            raise ValueError(f"all images must have one size; got {hw} and {size}")
        corners = find_checkerboard(arr, pattern)
        found.append(corners is not None)
        if corners is not None:
            obj_pts.append(objp)
            img_pts.append(corners)
    if size is None or len(img_pts) < min_views:
        raise ValueError(f"board found in {len(img_pts)} images; need >= {min_views}")
    # k3 fixed at 0: with a dozen webcam views it only overfits
    rms, k, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, size, np.eye(3), np.zeros(5), flags=cv2.CALIB_FIX_K3
    )
    d = np.asarray(dist, dtype=np.float64).ravel()[:5]
    intr = Intrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        dist=tuple(float(v) for v in d),
        width=size[0],
        height=size[1],
        rms_px=float(rms),
    )
    return intr, found
