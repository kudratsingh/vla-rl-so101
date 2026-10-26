"""Regenerate `so101.urdf` + `collision/*.stl` from the vendored upstream `so101_new_calib.urdf`.

Run once after re-vendoring:  uv run python src/env/assets/so101/build_collision.py

Why: PhysX treats every collision mesh as its convex hull. The hull of the fixed-jaw part
(`wrist_roll_follower_so101_v1.stl`) fills the space between the jaws, so cubes cannot be grasped.
We slice each jaw into slabs along the finger and hull each slab separately, and add a `tcp_link`
(grasp centre for a 3 cm object). Visual meshes are untouched.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from transforms3d.euler import euler2mat, mat2euler

HERE = Path(__file__).resolve().parent
SRC_URDF = HERE / "so101_new_calib.urdf"
OUT_URDF = HERE / "so101.urdf"
COL_DIR = HERE / "collision"

# Grasp centre in gripper_link frame: centre of a 3 cm cube clamped against the fixed-jaw tip,
# 13 mm above the fixed-jaw tip so the fingertips clear the table when grasping a cube on it.
TCP_XYZ = (0.0068, 0.0, -0.092)
# TCP axes in gripper_link: z = approach (out of the fingers, -z), y = closing direction (+x).
TCP_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _tf(xyz: list[float], rpy: list[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = euler2mat(*rpy, "sxyz")
    m[:3, 3] = xyz
    return m


def _plane_crossings(v: np.ndarray, edges: np.ndarray, h: np.ndarray, level: float) -> np.ndarray:
    """Points where mesh edges cross the plane `h == level`."""
    a, b = h[edges[:, 0]], h[edges[:, 1]]
    cross = (a - level) * (b - level) < 0
    e = edges[cross]
    t = (level - h[e[:, 0]]) / (h[e[:, 1]] - h[e[:, 0]])
    return np.asarray(v[e[:, 0]] + t[:, None] * (v[e[:, 1]] - v[e[:, 0]]))


def _slab_hulls(mesh: Any, axis: np.ndarray, cuts: list[float]) -> list[Any]:
    """Convex hulls of the mesh between consecutive planes `axis·p = cuts[i]`.

    Each slab = mesh vertices inside it + edge/plane crossing points on both faces (exact cut).
    """
    v = np.asarray(mesh.vertices)
    edges = np.asarray(mesh.edges_unique)
    h = v @ axis
    hulls = []
    for lo, hi in itertools.pairwise(cuts):
        pts = [v[(h >= lo) & (h <= hi)]]
        pts += [_plane_crossings(v, edges, h, lo), _plane_crossings(v, edges, h, hi)]
        cloud = np.concatenate(pts)
        if len(cloud) >= 4:
            hulls.append(_small_hull(cloud))
    return hulls


def _small_hull(cloud: np.ndarray, max_vertices: int = 64) -> Any:
    """Convex hull with <= max_vertices (PhysX caps convex meshes at 255): snap to a coarser grid."""
    hull = trimesh.convex.convex_hull(cloud)
    for grid in (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003):
        if len(hull.vertices) <= max_vertices:
            break
        snapped = np.unique(np.round(np.asarray(hull.vertices) / grid) * grid, axis=0)
        hull = trimesh.convex.convex_hull(snapped)
    return hull


def _origin(elem: ET.Element) -> tuple[list[float], list[float]]:
    o = elem.find("origin")
    assert o is not None
    return [float(v) for v in o.get("xyz", "0 0 0").split()], [
        float(v) for v in o.get("rpy", "0 0 0").split()
    ]


def _replace_collision(link: ET.Element, mesh_name: str, files: list[str]) -> None:
    for col in list(link.findall("collision")):
        mesh = col.find("geometry/mesh")
        if mesh is not None and mesh_name in mesh.get("filename", ""):
            link.remove(col)
    for f in files:
        col = ET.SubElement(link, "collision")
        ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(col, "geometry")
        ET.SubElement(geom, "mesh", filename=f"collision/{f}")


def _visual_tf(link: ET.Element, mesh_name: str) -> np.ndarray:
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is not None and mesh_name in mesh.get("filename", ""):
            return _tf(*_origin(vis))
    raise KeyError(mesh_name)


def main() -> None:
    COL_DIR.mkdir(exist_ok=True)
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}

    # Fixed jaw: part of gripper_link. Fingers point along -z of gripper_link.
    gl = links["gripper_link"]
    fixed = trimesh.load(HERE / "assets" / "wrist_roll_follower_so101_v1.stl")
    fixed.apply_transform(_visual_tf(gl, "wrist_roll_follower_so101_v1"))
    z = np.array([0.0, 0.0, 1.0])
    pieces = _slab_hulls(fixed, z, [-0.040, 0.005])  # body above the finger
    pieces += _slab_hulls(fixed, z, [-0.106, -0.093, -0.080, -0.067, -0.054, -0.040])
    fixed_files = []
    for i, p in enumerate(pieces):
        name = f"fixed_jaw_{i}.stl"
        p.export(COL_DIR / name)
        fixed_files.append(name)
    _replace_collision(gl, "wrist_roll_follower_so101_v1", fixed_files)

    # Moving jaw: in its own link frame the finger points along -y (tip at y ~ -0.082).
    mj = links["moving_jaw_so101_v1_link"]
    moving = trimesh.load(HERE / "assets" / "moving_jaw_so101_v1.stl")
    moving.apply_transform(_visual_tf(mj, "moving_jaw_so101_v1"))
    y = np.array([0.0, 1.0, 0.0])
    pieces = _slab_hulls(moving, y, [-0.0266, 0.011])  # plate near the pivot
    pieces += _slab_hulls(moving, y, [-0.083, -0.072, -0.061, -0.050, -0.038, -0.0266])
    moving_files = []
    for i, p in enumerate(pieces):
        name = f"moving_jaw_{i}.stl"
        p.export(COL_DIR / name)
        moving_files.append(name)
    _replace_collision(mj, "moving_jaw_so101_v1", moving_files)

    # TCP link (massless dummy, like upstream gripper_frame_link).
    tcp = ET.SubElement(root, "link", name="tcp_link")
    inertial = ET.SubElement(tcp, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1e-9")
    ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")
    joint = ET.SubElement(root, "joint", name="tcp_joint", type="fixed")
    rpy = mat2euler(TCP_ROT, "sxyz")
    ET.SubElement(
        joint,
        "origin",
        xyz=" ".join(f"{v:.6g}" for v in TCP_XYZ),
        rpy=" ".join(f"{v:.6g}" for v in rpy),
    )
    ET.SubElement(joint, "parent", link="gripper_link")
    ET.SubElement(joint, "child", link="tcp_link")

    root.set("name", "so101")
    root.insert(
        0,
        ET.Comment(
            " Generated by build_collision.py from so101_new_calib.urdf (TheRobotStudio/SO-ARM100,"
            " Apache-2.0). Edits: jaw collision meshes replaced by convex slabs; tcp_link added. "
        ),
    )
    ET.indent(tree, space="  ")
    tree.write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT_URDF} ({len(fixed_files)} fixed-jaw, {len(moving_files)} moving-jaw hulls)")


if __name__ == "__main__":
    main()
