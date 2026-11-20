"""Export Panda rigid-link visuals and joint metadata for Unreal Engine.

Uses only the Python standard library. The generated OBJ files use centimetres,
MuJoCo-oriented X/Y/Z coordinates, and one pivot per MJCF body. Unreal's OBJ
importer reflects Y on import, yielding Unreal-oriented X/-Y/Z mesh assets.
They are visual meshes; articulation and collision need separate Unreal setup.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ARM_XML = ROOT / "bas_lab" / "mod_panda_arm.xml"
SCENE_XML = ROOT / "bas_lab" / "bas_lab.xml"
ASSETS = ROOT / "mujoco_menagerie" / "franka_emika_panda" / "assets"
DEFAULT_OUT = ROOT / "unreal" / "exported" / "Panda"


def numbers(value: str, expected: int) -> list[float]:
    result = [float(x) for x in value.split()]
    if len(result) != expected:
        raise ValueError(f"Expected {expected} numbers, got {value!r}")
    return result


def ue_vector(v: list[float], scale: float = 1.0) -> list[float]:
    return [scale * v[0], -scale * v[1], scale * v[2]]


def ue_rotation(q: list[float]) -> list[list[float]]:
    """Return H R(q) H, where H=diag(1,-1,1) and MJCF q is w,x,y,z."""
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = (v / n for v in (w, x, y, z))
    r = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    h = (1, -1, 1)
    return [[h[i] * r[i][j] * h[j] for j in range(3)] for i in range(3)]


def obj_index(token: str, count: int, offset: int) -> str:
    if not token:
        return ""
    value = int(token)
    if value == 0:
        raise ValueError("OBJ indices are 1-based")
    return str(offset + (value if value > 0 else count + value + 1))


def emit_obj_part(out, source: Path, material: str, offsets: list[int]) -> list[int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    lines = source.read_text().splitlines()
    counts = [sum(line.startswith(prefix) for line in lines) for prefix in ("v ", "vt ", "vn ")]
    has_source_uv = counts[1] > 0
    if not has_source_uv:
        counts[1] = counts[0]
    out.write(f"usemtl {material}\n")
    for line in lines:
        if line.startswith("v "):
            v = numbers(line[2:], 3)
            # Unreal's Interchange OBJ translator performs X,Y,Z -> X,-Y,Z.
            # Keep source coordinates here so that reflection occurs once.
            p = [100.0 * component for component in v]
            out.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
            if not has_source_uv:
                # The Menagerie OBJs have no UVs. Interchange expects each
                # face vertex to reference a valid UV index even for solid colours.
                out.write(f"vt {(p[0] + p[2]) / 100:.8f} {(p[1] + p[2]) / 100:.8f}\n")
        elif line.startswith("vt "):
            out.write(line + "\n")
        elif line.startswith("vn "):
            v = numbers(line[3:], 3)
            out.write(f"vn {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for line in lines:
        if not line.startswith("f "):
            continue
        face = []
        for item in line[2:].split():
            pieces = item.split("/")
            if len(pieces) > 3:
                raise ValueError(f"Unsupported face entry: {item}")
            vertex = obj_index(pieces[0], counts[0], offsets[0])
            uv_source = pieces[1] if len(pieces) > 1 and pieces[1] else pieces[0]
            uv = obj_index(uv_source, counts[1], offsets[1])
            normal = obj_index(pieces[2], counts[2], offsets[2]) if len(pieces) > 2 and pieces[2] else ""
            face.append(f"{vertex}/{uv}/{normal}" if normal else f"{vertex}/{uv}")
        # The importer reflects Y, which reverses winding. Reverse the source
        # face here to keep imported normals and front faces consistent.
        out.write("f " + " ".join(reversed(face)) + "\n")
    return [a + b for a, b in zip(offsets, counts)]


def export(output_dir: Path) -> dict:
    arm = ET.parse(ARM_XML).getroot()
    mesh_files = {
        m.get("name", Path(m.attrib["file"]).stem): m.attrib["file"]
        for m in arm.findall("./asset/mesh")
    }
    materials = {
        m.attrib["name"]: numbers(m.attrib["rgba"], 4)
        for m in arm.findall("./asset/material")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "PandaMaterials.mtl").open("w") as out:
        out.write("# Colours from mod_panda_arm.xml\n")
        for name, rgba in materials.items():
            out.write(f"newmtl {name}\nKd {rgba[0]} {rgba[1]} {rgba[2]}\nd {rgba[3]}\n\n")

    bodies = []
    def walk(body: ET.Element, parent: str | None) -> None:
        name = body.attrib["name"]
        visual_geoms = [g for g in body.findall("geom") if g.get("class") == "visual"]
        collisions = [g for g in body.findall("geom") if g.get("class") != "visual"]
        visual_meshes = []
        offsets = [0, 0, 0]
        with (output_dir / f"{name}.obj").open("w") as obj:
            obj.write("# Panda link visual; local pivot at MJCF body origin; cm, source X/Y/Z\n")
            obj.write("mtllib PandaMaterials.mtl\n")
            obj.write(f"o {name}\n")
            for geom in visual_geoms:
                if geom.get("pos") or geom.get("quat") or geom.get("scale"):
                    raise ValueError(f"Visual geom transform needs explicit handling: {name}")
                mesh = geom.attrib["mesh"]
                material = geom.attrib["material"]
                source = ASSETS / mesh_files[mesh]
                offsets = emit_obj_part(obj, source, material, offsets)
                visual_meshes.append({"mesh": mesh, "source": str(source.relative_to(ROOT)), "material": material})

        joints = []
        for joint in body.findall("joint"):
            finger = joint.get("class") == "finger"
            kind = joint.get("type", "slide" if finger else "hinge")
            axis = numbers(joint.get("axis", "0 1 0" if finger else "0 0 1"), 3)
            limits = numbers(joint.get("range", "0 0.04" if finger else "-2.8973 2.8973"), 2)
            # Reflection changes the sign of an axial (rotation) vector, while
            # a slider direction transforms as an ordinary position vector.
            axis_ue = ue_vector(axis)
            if kind == "hinge":
                axis_ue = [-v for v in axis_ue]
            joints.append({
                "name": joint.attrib["name"], "type": kind,
                "axis_local_ue": axis_ue,
                "limits": [v * 100 if kind == "slide" else v for v in limits],
                "limit_unit": "cm" if kind == "slide" else "rad",
            })
        record = {
            "name": name, "parent": parent,
            "local_position_cm": ue_vector(numbers(body.get("pos", "0 0 0"), 3), 100.0),
            "local_rotation_matrix": ue_rotation(numbers(body.get("quat", "1 0 0 0"), 4)),
            "visual_obj": f"{name}.obj",
            "visual_parts": visual_meshes,
            "collision_sources": [
                {"geom": dict(g.attrib), "source_mesh": mesh_files[g.attrib["mesh"]] if "mesh" in g.attrib else None}
                for g in collisions
            ],
            "joints": joints,
            "mass_kg": float(body.find("inertial").attrib["mass"]),
        }
        bodies.append(record)
        for child in body.findall("body"):
            walk(child, name)

    roots = arm.findall("./worldbody/body")
    if len(roots) != 1:
        raise ValueError("Expected exactly one Panda root body")
    walk(roots[0], None)
    # BAS main XML has a MuJoCo-tolerated '--' in a comment, invalid for strict
    # ElementTree XML parsing. Read only the generated home keyframe attribute.
    match = re.search(r'<key\s+name="home"\s+qpos="([^"]+)"', SCENE_XML.read_text())
    if match is None:
        raise ValueError("Missing BAS home keyframe")
    home_qpos = [float(v) for v in match.group(1).split()]
    manifest = {
        "schema_version": 1,
        "source": str(ARM_XML.relative_to(ROOT)),
        "coordinates": "Unreal centimetres, X forward, Y=-MuJoCo Y, Z up; local rotations H*R*H",
        "home_arm_rad": home_qpos[:7],
        "home_finger_cm": [100 * v for v in home_qpos[7:9]],
        "materials_rgba": materials,
        "bodies": bodies,
        "notes": "Visual meshes only. Build joint constraints, drives and collision bodies separately in Unreal.",
    }
    (output_dir / "panda_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = export(args.output)
    print(f"Exported {len(manifest['bodies'])} rigid link OBJ files to {args.output}")
    print(f"Joints: {sum(len(b['joints']) for b in manifest['bodies'])}")


if __name__ == "__main__":
    main()
