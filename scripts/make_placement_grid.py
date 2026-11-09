#!/usr/bin/env python
"""WO-07: write docs/assets/placement-grid-a3.pdf, the 1:1 printed placement grid.

A3 portrait (297 x 420 mm), vector PDF written directly (no new dependency), deterministic bytes
(no dates), so a regenerated file diffs clean and the test can compare it with the committed one.
Content, all derived from code (``src/eval/real_grid.py``, which reads ``src/eval/protocol.py`` and
``src/env/tasks.py``; the robot base outline from the SO-101 URDF's ``base_link`` meshes):

* the 6 x 6 slot lattice: seen slots (inner 4 x 4, = SEEN_REGION, white) and shifted slots (the
  ring, shaded), each with a centre cross and the slot id that ``scripts/wo07_real_eval.py``
  prints, row letters and column numbers with their x / y in metres;
* the robot base outline (convex hull of the base_link visual meshes projected on the table) and
  the base_link origin, so the sheet can be aligned to the arm; +x / +y axis arrows;
* a 10 cm scale bar with 1 cm ticks: measure it after printing; if it is not 100 mm the print
  was scaled (print at 100% / "Actual size", never "Fit to page");
* protocol real-0.1 (``src/eval/protocol_real.py``): the five frozen SEEN slots (heavy outline,
  "SEEN"), the 11 other inner slots marked "not used" (never a demo, a camera-match frame or a
  trial in real-0.1), and the FIXED bowl. The bowl centre (x 0.195, y +0.205 m) lies 56.5 mm beyond
  the A3 sheet's left edge (the sheet cannot hold the whole 1:1 layout), so the sheet prints a
  "bowl line" at x = 0.195 m ending in a reference tick at y = +0.105 m: the bowl centre is exactly
  100 mm further left along that line. Mark it on the table once and keep the bowl on it.

    uv run scripts/make_placement_grid.py --preset real          # writes the PDF
    uv run scripts/make_placement_grid.py --preset real --check  # exit 1 if it is stale

``--config`` / ``--preset`` are accepted for the project convention and only ``real`` is allowed;
the grid reads no config: it is derived only from ``protocol.py`` and ``real_grid.py``, so
``--check`` stays deterministic.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args
from src.eval import protocol as P
from src.eval import protocol_real as PR
from src.eval import real_grid as G

OUT = REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf"
URDF = REPO_ROOT / "src" / "env" / "assets" / "so101" / "so101_new_calib.urdf"
PAGE_MM = (297.0, 420.0)
PT_PER_MM = 72.0 / 25.4
BASE_X_MM = PAGE_MM[0] / 2  # robot base origin on the page
BASE_Y_MM = 40.0  # base outline's back edge ~9 mm above the sheet's bottom edge


def page(x_m: float, y_m: float) -> tuple[float, float]:
    """Robot base frame (m; x forward, y left) -> page (mm from bottom-left; x up the page)."""
    return BASE_X_MM - y_m * 1000.0, BASE_Y_MM + x_m * 1000.0


# ------------------------------------------------------------------------------------------------
# Robot base outline from the URDF
# ------------------------------------------------------------------------------------------------
def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx  # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)


def _stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL")
    tri = np.frombuffer(
        data,
        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        count=n,
        offset=84,
    )
    return np.asarray(tri["v"], dtype=np.float64).reshape(-1, 3)


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    """2-D convex hull (Andrew's monotone chain), counter-clockwise."""
    pts = sorted({(round(float(x), 5), round(float(y), 5)) for x, y in points})

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def base_outline() -> list[tuple[float, float]]:
    """Convex hull (m, robot base frame) of the base_link visual meshes projected on the table."""
    root = ET.parse(URDF).getroot()
    link = next(lk for lk in root.iter("link") if lk.get("name") == "base_link")
    pts = []
    for vis in link.iter("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        origin = vis.find("origin")
        xyz = [
            float(v)
            for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        ]
        rpy = [
            float(v)
            for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        ]
        v = _stl_vertices(URDF.parent / str(mesh.get("filename")))
        pts.append(v @ _rpy_matrix(*rpy).T + np.asarray(xyz))
    allp = np.concatenate(pts)
    return _hull(allp[:, :2])


# ------------------------------------------------------------------------------------------------
# Minimal PDF writer
# ------------------------------------------------------------------------------------------------
def _f(v: float) -> str:
    return f"{v * PT_PER_MM:.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    """PDF content-stream operators, coordinates in mm."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def width(self, w_mm: float) -> None:
        self.ops.append(f"{_f(w_mm)} w")

    def gray(self, stroke: float, fill: float | None = None) -> None:
        self.ops.append(f"{stroke:.2f} G")
        if fill is not None:
            self.ops.append(f"{fill:.2f} g")

    def dash(self, on_mm: float = 0.0, off_mm: float = 0.0) -> None:
        self.ops.append(f"[{_f(on_mm)} {_f(off_mm)}] 0 d" if on_mm else "[] 0 d")

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(f"{_f(x0)} {_f(y0)} m {_f(x1)} {_f(y1)} l S")

    def rect(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        self.ops.append(f"{_f(x)} {_f(y)} {_f(w)} {_f(h)} re {'B' if fill else 'S'}")

    def poly(self, pts: Sequence[tuple[float, float]]) -> None:
        head, *rest = pts
        seg = " ".join(f"{_f(x)} {_f(y)} l" for x, y in rest)
        self.ops.append(f"{_f(head[0])} {_f(head[1])} m {seg} h S")

    def text(self, x: float, y: float, s: str, size_pt: float = 9.0, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        self.ops.append(f"BT /{font} {size_pt:.1f} Tf {_f(x)} {_f(y)} Td ({_esc(s)}) Tj ET")

    def arrow(self, x0: float, y0: float, x1: float, y1: float, head_mm: float = 3.0) -> None:
        self.line(x0, y0, x1, y1)
        a = math.atan2(y1 - y0, x1 - x0)
        for s in (+1, -1):
            b = a + math.pi - s * math.radians(25)
            self.line(x1, y1, x1 + head_mm * math.cos(b), y1 + head_mm * math.sin(b))


def pdf_bytes(content: str) -> bytes:
    w, h = (f"{v * PT_PER_MM:.2f}" for v in PAGE_MM)
    stream = content.encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /Font << "
            "/F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# ------------------------------------------------------------------------------------------------
# The sheet
# ------------------------------------------------------------------------------------------------
def draw() -> str:
    c = Canvas()
    slots = G.all_slots()
    dx, dy = G.cell_size_m()

    seen_ids = set(PR.SEEN_SLOTS)
    # slot rectangles: shifted shaded, seen white; centre crosses and ids
    for s in slots:
        x0, y0 = page(s.x - s.half_x, s.y + s.half_y)  # page bottom-left: min x, max y
        shaded = s.cell == "shifted"
        c.width(0.25)
        c.dash()
        c.gray(0.45, 0.88 if shaded else 1.0)
        c.rect(x0, y0, dy * 1000.0, dx * 1000.0, fill=True)
        px, py = page(s.x, s.y)
        c.gray(0.0, 0.0)
        c.width(0.35)
        c.line(px - 3, py, px + 3, py)
        c.line(px, py - 3, px, py + 3)
        c.text(px + 1.2, py + 1.6, s.id, 8.0, bold=True)
        if s.id in seen_ids:  # real-0.1 seen cell: heavy outline
            c.width(1.2)
            c.rect(x0 + 1.0, y0 + 1.0, dy * 1000.0 - 2.0, dx * 1000.0 - 2.0)
            c.text(px - 8.0, py - 6.0, "SEEN", 7.0, bold=True)
        elif s.cell == "seen":  # the other 11 inner slots: never used in real-0.1
            c.text(px - 8.0, py - 6.0, "not used", 6.0)

    # region borders: seen solid thick, shifted dashed
    for region, dashed in ((P.SEEN_REGION, False), (P.SHIFTED_REGION, True)):
        (cx, cy), (hx, hy) = region.center, region.half_extent
        x0, y0 = page(cx - hx, cy + hy)
        c.gray(0.0)
        c.width(0.5 if dashed else 0.8)
        c.dash(2.0, 1.5) if dashed else c.dash()
        c.rect(x0, y0, 2 * hy * 1000.0, 2 * hx * 1000.0)
    c.dash()

    # row / column labels with coordinates
    (scx, scy), (shx, shy) = P.SHIFTED_REGION.center, P.SHIFTED_REGION.half_extent
    left_x, _ = page(0.0, scy + shy)
    for r in range(G.N_ROWS):
        s = slots[r * G.N_COLS]
        _, py = page(s.x, 0.0)
        c.text(left_x - 24.0, py - 1.2, f"{G.ROW_LETTERS[r]}  x={s.x:.4f}", 7.5)
    for col in range(G.N_COLS):
        s = slots[col]
        px, _ = page(0.0, s.y)
        _, top = page(scx + shx, 0.0)
        c.text(px - 7.0, top + 3.0, f"{col + 1}  y={s.y:+.4f}", 7.0)

    # robot base outline + origin + axes
    hull = [page(x, y) for x, y in base_outline()]
    c.gray(0.0)
    c.width(0.6)
    c.poly(hull)
    ox, oy = page(0.0, 0.0)
    c.width(0.3)
    c.line(ox - 5, oy, ox + 5, oy)
    c.line(ox, oy - 5, ox, oy + 5)
    for i, line in enumerate(
        (
            "Robot base outline: URDF base_link",
            "meshes projected on the table;",
            "+ = base_link origin (robot frame).",
            "Set the arm's base on this outline,",
            "the sheet's long axis along +x.",
        )
    ):
        c.text(214.0, 100.0 - 4.2 * i, line, 7.5)
    c.width(0.5)
    c.arrow(ox, oy + 6, ox, oy + 80)
    c.text(ox + 2.0, oy + 76.0, "+x (forward)", 8.0, bold=True)
    c.arrow(ox - 6, oy, ox - 75, oy)
    c.text(ox - 78.0, oy + 2.0, "+y (robot's left)", 8.0, bold=True)

    # real-0.1 fixed bowl: a line at its x, a reference tick at y = +0.105 m (the grid's +y edge);
    # the bowl centre is (bowl y - 0.105) m further along the line, beyond the sheet's left edge
    bx, by = PR.BOWL_XY_M
    ref_y = P.SHIFTED_REGION.center[1] + P.SHIFTED_REGION.half_extent[1]
    lx, ly = page(bx, ref_y)
    c.gray(0.0)
    c.width(0.5)
    c.dash(3.0, 1.5)
    c.line(lx, ly, 0.0, ly)
    c.dash()
    c.width(0.8)
    c.line(lx, ly - 4.0, lx, ly + 4.0)
    off_mm = (by - ref_y) * 1000.0
    edge_mm = -page(bx, by)[0]
    c.text(2.0, ly + 2.0, "BOWL LINE", 6.0, bold=True)
    for i, line in enumerate(
        (
            (
                f"BOWL, FIXED for every demo and trial ({PR.PROTOCOL_VERSION}): centre x = "
                f"{bx:.3f} m, y = {by:+.3f} m."
            ),
            (
                f"Along the dashed BOWL LINE (left, x = {bx:.3f} m) the centre is {off_mm:.1f} mm "
                f"left of the line's end tick (y = {ref_y:+.3f} m),"
            ),
            (
                f"{edge_mm:.1f} mm beyond this sheet's left edge. Mark it on the table once; the "
                "bowl stays on the mark."
            ),
        )
    ):
        c.text(14.0, 345.0 - 4.2 * i, line, 7.5, bold=i == 0)

    # 10 cm scale bar
    sx, sy = 14.0, 360.0
    c.width(0.6)
    c.line(sx, sy, sx + 100.0, sy)
    for i in range(11):
        t = 3.0 if i in (0, 5, 10) else 1.8
        c.line(sx + 10.0 * i, sy, sx + 10.0 * i, sy + t)
    c.text(sx, sy + 4.5, "10 cm scale bar: must measure 100 mm", 8.0, bold=True)
    c.text(sx, sy - 5.0, "If not, reprint at 100% / Actual size (never Fit to page)", 7.0)

    # title block
    c.gray(0.0, 0.0)
    c.text(14.0, 406.0, "vla-rl-so101  WO-07 real-eval placement grid  (A3, 1:1)", 13.0, bold=True)
    c.text(
        14.0,
        399.0,
        "Place each object's CENTRE on its slot cross. Slot ids = scripts/wo07_real_eval.py prompts.",
        8.5,
    )
    c.text(
        14.0,
        394.0,
        f"White = seen / unseen_objects slots (SEEN_REGION centre {P.SEEN_REGION.center} m, half "
        f"{P.SEEN_REGION.half_extent} m); shaded = shifted (SHIFTED_REGION x{P.SHIFT_SCALE}, "
        "outside seen).",
        7.5,
    )
    c.text(
        14.0,
        389.5,
        f"Generated by scripts/make_placement_grid.py from src/eval/real_grid.py; src/eval/protocol.py "
        f"{P.PROTOCOL_VERSION} sha256 {P.protocol_hash()[:12]}. Robot base frame, metres.",
        7.5,
    )
    c.text(
        14.0,
        385.0,
        f"Slots: {dx * 100:.1f} cm (x) by {dy * 100:.1f} cm (y) rectangles of the shifted region; "
        "row letter = x (A nearest the base), column = y (1 = robot's left).",
        7.5,
    )
    c.text(
        14.0,
        380.5,
        f"Protocol {PR.PROTOCOL_VERSION} {PR.protocol_hash()[:12]}: SEEN = {' '.join(PR.SEEN_SLOTS)} "
        "(demos + seen + unseen-object trials); shifted = the shaded ring; 'not used' = never "
        "in real-0.1; bowl FIXED (bowl line).",
        7.5,
    )
    return "\n".join(c.ops) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("the placement grid is a real-arm artefact: --preset real")
    data = pdf_bytes(draw())
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != data:
            print(f"{args.out} is stale: run scripts/make_placement_grid.py --preset real")
            return 1
        print(f"{args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
