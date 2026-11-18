"""
build_bas_lab.py -- generates the MICROGRAVITY "Bharatiya Antariksh Station (BAS) laboratory" workcell for MuJoCo.

    .\\.venv\\Scripts\\python.exe build_bas_lab.py                # one cube + one target (default)
    .\\.venv\\Scripts\\python.exe build_bas_lab.py --extra_objects   # also writes bas_lab_extra_objects.xml (3 objects)

Output folder: bas_lab/   (open bas_lab/bas_lab.xml in MuJoCo; the older table scenes are NOT touched)

    bas_lab.xml                 main file: compiler, physics (gravity 0), visual, keyframe, includes the modules below
    mod_panda_arm.xml           Franka Panda (same body/joint/actuator names as the old scenes)
    mod_bas_module.xml          cylindrical pressurized module: shell, rack ring, end caps, hatch, porthole, flag, lights
    mod_workstation.xml         robot mounting rack, lab bench, velcro work surface, yellow target marker, red cube
    mod_cameras.xml             policy_cam (= the camera used for all training data), overview_cam, bench_cam, aisle_cam
    mod_extra_objects.xml       (only with --extra_objects) canister + small box
    textures/*.png              generated pictures (flag, title plate, Earth view, breadboard surface)

Coordinate system (same as the table scenes, so scripts/policies need no change):
    origin = robot base, +x forward over the bench, +y left, +z = module "up" (rack mounting direction).
    bench top z = 0.375; cube/marker ranges unchanged. Module axis runs along y through (x=0.55, z=0.85).
Microgravity: gravity = 0. Objects are held by a VELCRO work surface (MuJoCo adhesion actuator on the bench top) and
a little air drag (joint damping on the cube). The gripper pulls the cube off the velcro and presses it back on.
Everything that is not the bench/robot mount is visual only (no collisions) to keep the simulation fast.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bas_lab"
TEX = OUT / "textures"
SRC = ROOT / "mujoco_menagerie" / "franka_emika_panda" / "panda_pick_place_marker.xml"

# ---- module geometry (metres) --------------------------------------------------------------------------------
AX, AZ = 0.55, 0.85            # module axis passes through (x=AX, z=AZ), runs along y
A_IN = 1.90                    # distance axis -> inner shell panels
L_HALF = 3.5                   # module half-length (7 m long)
N_SIDES = 24                   # polygon approximation of the cylinder
RACK_D = 0.45                  # rack depth (radial)
RACK_W, RACK_H = 0.68, 0.88    # rack front: width (tangential) x length (along the module axis)
COL_Y = [-2.7, -1.8, -0.9, 0.0, 0.9, 1.8, 2.7]
SECTORS = [15 + 30 * k for k in range(12)]      # degrees, measured in the x-z plane from +x towards +z

NC = 'contype="0" conaffinity="0"'              # visual-only geoms
SAFFRON, WHITE, GREEN, NAVY = (255, 153, 51), (255, 255, 255), (19, 136, 8), (0, 0, 128)


def f(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def vec(*v):
    return " ".join(f(float(x)) for x in v)


def quat_from_axes(x, y, z):
    m = np.stack([x, y, z], axis=1).astype(float)          # columns = local axes in world coordinates
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, m.flatten())
    return q


def frame_for_sector(phi_deg):
    """Local frame of a rack at angle phi: z = inward normal (towards the module axis), y = module axis, x = tangent."""
    p = math.radians(phi_deg)
    z = np.array([-math.cos(p), 0.0, -math.sin(p)])
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return x, y, z


# ---- textures --------------------------------------------------------------------------------------------------
def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def make_textures():
    TEX.mkdir(parents=True, exist_ok=True)
    # tricolour banner with 24-spoke chakra
    w, h = 1024, 512
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, h // 3], fill=SAFFRON)
    d.rectangle([0, 2 * h // 3, w, h], fill=GREEN)
    cx, cy, r = w // 2, h // 2, h // 6 - 4
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=NAVY, width=5)
    for k in range(24):
        a = 2 * math.pi * k / 24
        d.line([cx, cy, cx + r * math.cos(a), cy + r * math.sin(a)], fill=NAVY, width=2)
    im.save(TEX / "banner.png")
    # title plate
    w, h = 1024, 256
    im = Image.new("RGB", (w, h), (12, 22, 48))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 10], fill=SAFFRON)
    d.rectangle([0, h - 10, w, h], fill=GREEN)
    f1 = font("C:/Windows/Fonts/arialbd.ttf", 50)
    f2 = font("C:/Windows/Fonts/Nirmala.ttc", 62)
    f3 = font("C:/Windows/Fonts/arial.ttf", 30)
    for text, ft, y in [("BHARATIYA ANTARIKSH STATION", f1, 22),
                        ("\u092d\u093e\u0930\u0924\u0940\u092f \u0905\u0902\u0924\u0930\u093f\u0915\u094d\u0937 \u0938\u094d\u091f\u0947\u0936\u0928", f2, 98),
                        ("SCIENCE & ROBOTICS LAB MODULE", f3, 195)]:
        tw = d.textlength(text, font=ft)
        d.text(((w - tw) / 2, y), text, font=ft, fill=(235, 240, 250))
    im.save(TEX / "title.png")
    # Earth through the porthole
    random.seed(7)
    s = 512
    im = Image.new("RGB", (s, s), (3, 6, 18))
    d = ImageDraw.Draw(im)
    for _ in range(120):
        d.point((random.randrange(s), random.randrange(s)), fill=(255, 255, 255))
    R = int(s * 0.95)
    cx, cy = s // 2, int(s * 1.05)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(24, 82, 170))
    for _ in range(14):
        x, y = random.randrange(40, s - 40), random.randrange(int(s * 0.45), s)
        rr = random.randrange(18, 55)
        d.ellipse([x - rr, y - rr // 2, x + rr, y + rr // 2], fill=(60, 120, 60))
    for _ in range(22):
        x, y = random.randrange(20, s - 20), random.randrange(int(s * 0.4), s)
        d.ellipse([x - 40, y - 8, x + 40, y + 8], fill=(235, 240, 250))
    d.arc([cx - R - 6, cy - R - 6, cx + R + 6, cy + R + 6], 200, 340, fill=(120, 190, 255), width=8)
    im.save(TEX / "earth.png")
    # velcro / optical-breadboard work surface: dark grey with a faint hole grid
    s = 512
    im = Image.new("RGB", (s, s), (58, 63, 72))
    d = ImageDraw.Draw(im)
    for i in range(8, s, 32):
        for j in range(8, s, 32):
            d.ellipse([i - 3, j - 3, i + 3, j + 3], fill=(36, 40, 47))
    d.rectangle([0, 0, s - 1, s - 1], outline=(80, 86, 96), width=2)
    im.save(TEX / "breadboard.png")


# ---- robot module ----------------------------------------------------------------------------------------------
def write_panda_module():
    src = SRC.read_text()
    default = src[src.index("<default>"): src.index("<asset>")].rstrip()
    assets = src[src.index("<asset>") + len("<asset>"): src.index("</asset>")]
    assets = "\n".join(l for l in assets.splitlines() if "marker_mat" not in l)
    body0 = src.index('<body name="link0" childclass="panda">')
    body1 = src.index("<!-- Ground plane -->")
    arm = src[body0:body1].rstrip()
    tail = src[src.index("<tendon>"): src.index("</actuator>") + len("</actuator>")]
    contact = src[src.index("<contact>"): src.index("</contact>") + len("</contact>")]
    xml = f"""<mujoco model="panda_arm_module">
  <!-- Franka Panda copied from the old scenes (identical names/limits/gains). It is mounted at the world origin.
       Further arms: use <attach> with a prefix (see bas_lab/README.md). -->
  {default}
  <asset>{assets}
  </asset>
  <worldbody>
    {arm}
  </worldbody>
  {tail}
  {contact}
</mujoco>
"""
    (OUT / "mod_panda_arm.xml").write_text(xml)


# ---- rack decoration -------------------------------------------------------------------------------------------
LOCKERS = ["locker_a", "locker_b", "locker_c"]


def g_box(name, pos, half, mat, extra=""):
    return f'<geom name="{name}" type="box" pos="{vec(*pos)}" size="{vec(*half)}" material="{mat}" {NC} {extra}/>'


def decorate_face(prefix, kind, W, H, rng):
    """Geoms in the rack-local frame: x across (W), y along (H), z = out of the face (face at z=0)."""
    out = []
    n = 0

    def add(pos, half, mat):
        nonlocal n
        out.append(g_box(f"{prefix}_{n}", pos, half, mat))
        n += 1

    if kind == "lockers":
        cols = max(2, round(W / 0.3))
        rows = max(2, round(H / 0.24))
        cw, rh = W / cols, H / rows
        for i in range(cols):
            for j in range(rows):
                x = -W / 2 + cw * (i + 0.5)
                y = -H / 2 + rh * (j + 0.5)
                add((x, y, 0.008), (cw / 2 - 0.012, rh / 2 - 0.012, 0.008), rng.choice(LOCKERS))
                add((x, y - rh * 0.28, 0.02), (0.035, 0.006, 0.006), "dark")
                if rng.random() < 0.5:
                    add((x, y + rh * 0.32, 0.017), (cw / 2 - 0.04, 0.007, 0.002), rng.choice(["accent_teal", "accent_blue"]))
    elif kind == "drawers":
        rows = max(3, round(H / 0.17))
        rh = H / rows
        for j in range(rows):
            y = -H / 2 + rh * (j + 0.5)
            add((0, y, 0.008), (W / 2 - 0.015, rh / 2 - 0.01, 0.008), rng.choice(LOCKERS))
            add((0, y, 0.02), (0.13, 0.008, 0.008), "dark")
            add((-W / 2 + 0.05, y + rh * 0.28, 0.017), (0.008, 0.008, 0.002), rng.choice(["led_green", "led_blue"]))
    elif kind == "display":
        add((0, H * 0.18, 0.006), (W / 2 - 0.04, H * 0.26, 0.006), "dark")
        add((0, H * 0.18, 0.013), (W / 2 - 0.07, H * 0.26 - 0.03, 0.002), "screen")
        add((0, -H * 0.2, 0.006), (W / 2 - 0.1, H * 0.14, 0.006), "dark")
        add((0, -H * 0.2, 0.013), (W / 2 - 0.12, H * 0.14 - 0.02, 0.002), "screen_dim")
        for i in range(5):
            add((-W / 2 + 0.1 + i * (W - 0.2) / 4, -H * 0.42, 0.01), (0.016, 0.016, 0.01), "dark")
            add((-W / 2 + 0.1 + i * (W - 0.2) / 4, -H * 0.36, 0.012), (0.007, 0.007, 0.002), rng.choice(["led_green", "led_blue"]))
    else:  # vent: grille panels and status lights
        add((0, 0, 0.006), (W / 2 - 0.03, H / 2 - 0.03, 0.006), "locker_b")
        for j in range(9):
            add((0, -H / 2 + 0.08 + j * (H - 0.16) / 8, 0.014), (W / 2 - 0.08, 0.01, 0.004), "dark")
        add((-W / 2 + 0.06, H / 2 - 0.05, 0.016), (0.012, 0.012, 0.003), "led_green")
        add((W / 2 - 0.06, H / 2 - 0.05, 0.016), (0.012, 0.012, 0.003), "led_blue")
    return out


def rack_body(prefix, W, H, D):
    return g_box(f"{prefix}_body", (0, 0, -D / 2), (W / 2, H / 2, D / 2), "rack_body")


def make_module(rng):
    g = []   # worldbody children
    r_shell = A_IN + 0.025
    chord = 2 * A_IN * math.tan(math.pi / N_SIDES)
    # shell panels
    for k in range(N_SIDES):
        phi = 360.0 / N_SIDES * k
        x, y, z = frame_for_sector(phi)
        q = quat_from_axes(x, y, z)
        c = np.array([AX, 0, AZ]) + r_shell * np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
        g.append(f'<geom name="shell_{k}" type="box" pos="{vec(*c)}" quat="{vec(*q)}" '
                 f'size="{vec(chord / 2 + 0.01, L_HALF, 0.025)}" material="shell_mat" {NC}/>')
    # end caps (inner face at y = +-L_HALF) with structure
    for sgn, nm in [(1, "far"), (-1, "near")]:
        g.append(f'<geom name="cap_{nm}" type="cylinder" pos="{vec(AX, sgn * (L_HALF + 0.03), AZ)}" euler="1.5708 0 0" '
                 f'size="{vec(A_IN + 0.05, 0.03)}" material="shell_mat" {NC}/>')
    yc = L_HALF
    # far cap: hatch, title plate, flag
    g.append(f'<geom name="hatch_ring" type="cylinder" pos="{vec(AX, yc - 0.03, AZ)}" euler="1.5708 0 0" size="0.62 0.03" material="frame_mat" {NC}/>')
    g.append(f'<geom name="hatch_door" type="cylinder" pos="{vec(AX, yc - 0.06, AZ)}" euler="1.5708 0 0" size="0.5 0.02" material="locker_b" {NC}/>')
    g.append(f'<geom name="hatch_window" type="cylinder" pos="{vec(AX, yc - 0.083, AZ + 0.18)}" euler="1.5708 0 0" size="0.14 0.004" material="screen_dim" {NC}/>')
    g.append(f'<geom name="hatch_wheel" type="cylinder" pos="{vec(AX, yc - 0.09, AZ - 0.12)}" euler="1.5708 0 0" size="0.13 0.012" material="accent_blue" {NC}/>')
    g.append(f'<geom name="hatch_hub" type="cylinder" pos="{vec(AX, yc - 0.105, AZ - 0.12)}" euler="1.5708 0 0" size="0.035 0.015" material="dark" {NC}/>')
    g.append(f'<geom name="cap_title" type="plane" pos="{vec(AX, yc - 0.012, AZ + 1.0)}" size="0.62 0.155 0.01" xyaxes="1 0 0 0 0 1" material="title_mat" {NC}/>')
    g.append(f'<geom name="cap_flag" type="plane" pos="{vec(AX, -yc + 0.012, AZ + 0.98)}" size="0.33 0.22 0.01" xyaxes="-1 0 0 0 0 1" material="banner_mat" {NC}/>')
    g.append(f'<geom name="cap_flag_frame" type="box" pos="{vec(AX, -yc + 0.005, AZ + 0.98)}" size="0.35 0.004 0.24" material="frame_mat" {NC}/>')
    # near cap: plain hatch so the module looks closed at both ends
    g.append(f'<geom name="hatch_ring_n" type="cylinder" pos="{vec(AX, -yc + 0.03, AZ)}" euler="1.5708 0 0" size="0.62 0.03" material="frame_mat" {NC}/>')
    g.append(f'<geom name="hatch_door_n" type="cylinder" pos="{vec(AX, -yc + 0.06, AZ)}" euler="1.5708 0 0" size="0.5 0.02" material="locker_b" {NC}/>')
    # ring of racks
    rack_front_r = A_IN - RACK_D
    count = 0
    for phi in SECTORS:
        x, y, z = frame_for_sector(phi)
        q = quat_from_axes(x, y, z)
        for yc_ in COL_Y:
            if phi in (255, 285) and abs(yc_) <= 1.0:
                continue                                    # space for the robot mounting rack and the lab bench
            if phi == 105 and yc_ == 1.8:
                continue                                    # porthole sits here (below)
            c = np.array([AX, yc_, AZ]) + rack_front_r * np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
            kind = rng.choice(["lockers", "drawers", "display", "vent", "lockers", "display"])
            pre = f"rk{count}"
            geoms = [rack_body(pre, RACK_W, RACK_H, RACK_D)] + decorate_face(pre, kind, RACK_W, RACK_H, rng)
            g.append(f'<body name="{pre}" pos="{vec(*c)}" quat="{vec(*q)}">\n      ' + "\n      ".join(geoms) + "\n    </body>")
            count += 1
    # porthole to Earth
    phi = 105
    x, y, z = frame_for_sector(phi)
    q = quat_from_axes(x, y, z)
    c = np.array([AX, 1.8, AZ]) + (A_IN - 0.02) * np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
    g.append(f'<body name="porthole" pos="{vec(*c)}" quat="{vec(*q)}">'
             f'<geom name="porthole_ring" type="cylinder" pos="0 0 0.0" size="0.42 0.025" material="frame_mat" {NC}/>'
             f'<geom name="porthole_view" type="plane" pos="0 0 0.03" size="0.3 0.3 0.01" material="earth_mat" {NC}/></body>')
    # LED light strips and handrails along the seams between rack sectors
    for k, phi in enumerate(range(0, 360, 30)):
        if phi in (270,):
            continue
        x, y, z = frame_for_sector(phi)
        q = quat_from_axes(x, y, z)
        c = np.array([AX, 0, AZ]) + (rack_front_r + 0.01) * np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
        mat = "led_white" if phi in (60, 90, 120, 240, 300) else "frame_mat"
        g.append(f'<body name="seam{k}" pos="{vec(*c)}" quat="{vec(*q)}">'
                 f'<geom name="seam_strip{k}" type="box" pos="0 0 0" size="0.012 {L_HALF - 0.4} 0.006" material="{mat}" {NC}/></body>')
    for k, phi in enumerate([0, 180]):
        x, y, z = frame_for_sector(phi)
        q = quat_from_axes(x, y, z)
        c = np.array([AX, 0, AZ]) + (rack_front_r - 0.05) * np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
        g.append(f'<body name="rail{k}" pos="{vec(*c)}" quat="{vec(*q)}">'
                 f'<geom name="handrail{k}" type="capsule" fromto="0 {-(L_HALF - 0.6)} 0 0 {L_HALF - 0.6} 0" size="0.02" material="rail_mat" {NC}/></body>')

    assets = """
  <asset>
    <texture name="shell_tex" type="2d" builtin="checker" rgb1="0.58 0.60 0.65" rgb2="0.56 0.58 0.63" width="512" height="512" mark="edge" markrgb="0.45 0.48 0.55"/>
    <texture name="banner_tex" type="2d" file="banner.png"/>
    <texture name="title_tex" type="2d" file="title.png"/>
    <texture name="earth_tex" type="2d" file="earth.png"/>
    <material name="shell_mat" texture="shell_tex" texrepeat="2 2" texuniform="true" specular="0.2"/>
    <material name="banner_mat" texture="banner_tex" emission="0.35" specular="0"/>
    <material name="title_mat" texture="title_tex" emission="0.5" specular="0"/>
    <material name="earth_mat" texture="earth_tex" emission="0.8" specular="0"/>
    <material name="rack_body" rgba="0.46 0.49 0.54 1" specular="0.3"/>
    <material name="locker_a" rgba="0.70 0.72 0.76 1" specular="0.3"/>
    <material name="locker_b" rgba="0.62 0.66 0.71 1" specular="0.3"/>
    <material name="locker_c" rgba="0.60 0.67 0.64 1" specular="0.3"/>
    <material name="dark" rgba="0.12 0.13 0.16 1" specular="0.2"/>
    <material name="accent_teal" rgba="0.10 0.55 0.60 1"/>
    <material name="accent_blue" rgba="0.15 0.35 0.75 1" specular="0.4"/>
    <material name="screen" rgba="0.08 0.38 0.55 1" emission="0.45" specular="0"/>
    <material name="screen_dim" rgba="0.04 0.18 0.30 1" emission="0.3" specular="0"/>
    <material name="led_green" rgba="0.1 1.0 0.3 1" emission="1" specular="0"/>
    <material name="led_blue" rgba="0.2 0.6 1.0 1" emission="1" specular="0"/>
    <material name="led_white" rgba="1 1 1 1" emission="1" specular="0"/>
    <material name="frame_mat" rgba="0.48 0.50 0.56 1" specular="0.6" shininess="0.6"/>
    <material name="rail_mat" rgba="0.15 0.45 0.85 1" specular="0.6" shininess="0.5"/>
  </asset>
"""
    xml = f"""<mujoco model="bas_module">
  <!-- Pressurized cylindrical module, axis along y through (x={AX}, z={AZ}), inner radius {A_IN} m, length {2 * L_HALF} m.
       24-sided shell, ring of {count} equipment/storage racks, end caps with hatch/title/flag, Earth porthole.
       Visual only (no collisions). Generated by build_bas_lab.py -->
{assets}
  <worldbody>
    {chr(10).join('    ' + s for s in g).strip()}
  </worldbody>
</mujoco>
"""
    (OUT / "mod_bas_module.xml").write_text(xml)
    return count


def make_workstation(rng, extra):
    D = -1.03       # deck level under the bench
    faces = []

    def face_body(name, pos, zdir, ydir, W, H, kind):
        z = np.array(zdir, float)
        y = np.array(ydir, float)
        x = np.cross(y, z)
        q = quat_from_axes(x, y, z)
        geoms = decorate_face(name, kind, W, H, rng)
        faces.append(f'<body name="{name}" pos="{vec(*pos)}" quat="{vec(*q)}">\n      ' + "\n      ".join(geoms) + "\n    </body>")

    h_b = 0.325 - D
    h_p = 0.0 - D
    # robot mounting rack (pedestal, x -0.30..0.20) and lab bench rack (x 0.20..0.90), y +-0.40
    face_body("ped_face_ny", (-0.05, -0.4, D + h_p / 2), (0, -1, 0), (0, 0, 1), 0.5, h_p - 0.1, "lockers")
    face_body("ped_face_py", (-0.05, 0.4, D + h_p / 2), (0, 1, 0), (0, 0, 1), 0.5, h_p - 0.1, "drawers")
    face_body("ped_face_nx", (-0.3, 0.0, D + h_p / 2), (-1, 0, 0), (0, 0, 1), 0.8, h_p - 0.1, "vent")
    face_body("bench_face_ny", (0.55, -0.4, D + h_b / 2), (0, -1, 0), (0, 0, 1), 0.7, h_b - 0.1, "display")
    face_body("bench_face_px", (0.9, 0.0, D + h_b / 2), (1, 0, 0), (0, 0, 1), 0.8, h_b - 0.1, "lockers")
    face_body("bench_face_py", (0.55, 0.4, D + h_b / 2), (0, 1, 0), (0, 0, 1), 0.7, h_b - 0.1, "drawers")

    extra_bodies = ""
    if extra:
        extra_bodies = ""

    xml = f"""<mujoco model="bas_workstation">
  <!-- Robot mounting rack + lab bench. Bench top (z=0.375) = same height/size as the old table, so cube/marker ranges,
       robot reach and the policy camera view are unchanged. The bench top is a VELCRO surface: the adhesion actuator
       "velcro_surface" pulls any object touching it onto the surface (microgravity holding mechanism). -->
  <asset>
    <texture name="bb_tex" type="2d" file="breadboard.png"/>
    <material name="bench_mat" texture="bb_tex" texrepeat="3 3" texuniform="true" specular="0.2" reflectance="0.05"/>
    <material name="marker_mat" rgba="1.0 0.85 0.0 1" emission="1.0" specular="0" shininess="0"/>
    <material name="mount_mat" rgba="0.30 0.33 0.38 1" specular="0.5"/>
  </asset>
  <worldbody>
    <!-- robot mounting rack: the arm is bolted on top of it at the world origin -->
    <geom name="mount_rack" type="box" pos="-0.05 0 {D + h_p / 2:.4f}" size="0.25 0.40 {h_p / 2:.4f}" material="rack_body"/>
    <geom name="mount_plate" type="box" pos="0 0 -0.015" size="0.20 0.20 0.015" material="mount_mat" {NC}/>
    <geom name="mount_flange" type="cylinder" pos="0 0 0.004" size="0.075 0.004" material="frame_mat" {NC}/>
    <!-- lab bench rack -->
    <geom name="bench_rack" type="box" pos="0.55 0 {D + h_b / 2:.4f}" size="0.35 0.40 {h_b / 2:.4f}" material="rack_body"/>
    <body name="bench_top" pos="0 0 0">
      <geom name="table_top" type="box" pos="0.55 0 0.35" size="0.35 0.40 0.025" material="bench_mat" friction="1.2 0.02 0.002"/>
    </body>
    {chr(10).join('    ' + s for s in faces).strip()}

    <!-- Visible placement target: flat 10x10 cm self-lit yellow square (visual only). Mocap body, as in the old scenes. -->
    <body name="target_marker" mocap="true" pos="0.60 0.0 0.3765">
      <geom name="target_marker_geom" type="box" size="0.05 0.05 0.0005" material="marker_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Free cube: 5 cm per side, 80 grams. Small damping = air drag in the cabin so nothing drifts forever. -->
    <body name="pick_cube" pos="0.45 0 0.40">
      <joint name="cube_free" type="free" damping="0.01"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.9 0.12 0.08 1" friction="1.5 0.02 0.002" margin="0.003" gap="0.003"/>
    </body>
    {extra_bodies}
  </worldbody>
  <actuator>
    <!-- velcro: pulls objects that touch the bench top onto it. ctrl=1 means on (set by the keyframe); gripper can overcome it -->
    <adhesion name="velcro_surface" body="bench_top" ctrlrange="0 1" gain="1.5"/>
  </actuator>
</mujoco>
"""
    (OUT / "mod_workstation.xml").write_text(xml)


def make_cameras():
    # policy_cam reproduces the MjvCamera used for all training data: lookat (0.55,0,0.43), distance 1.65, az 135, el -25
    az, el = math.radians(135), math.radians(-25)
    fwd = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    pos = np.array([0.55, 0.0, 0.43]) - 1.65 * fwd
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    def look(p, target, fovy):
        p, target = np.array(p, float), np.array(target, float)
        fw = target - p; fw /= np.linalg.norm(fw)
        rt = np.cross(fw, [0, 0, 1]); rt /= np.linalg.norm(rt)
        u = np.cross(rt, fw)
        return f'pos="{vec(*p)}" xyaxes="{vec(*rt, *u)}" fovy="{fovy}"'

    xml = f"""<mujoco model="bas_cameras_lights">
  <!-- Cameras are fixed in the world frame. policy_cam == the camera of the data/evaluation scripts (free camera, fovy 45). -->
  <worldbody>
    <camera name="policy_cam" pos="{vec(*pos)}" xyaxes="{vec(*right, *up)}" fovy="45"/>
    <camera name="overview_cam" {look((0.55, -3.1, 1.4), (0.55, 0.5, 0.5), 70)}/>
    <camera name="bench_cam" {look((1.5, 0.0, 1.0), (0.55, 0.0, 0.4), 50)}/>
    <camera name="end_cam" {look((0.55, 3.1, 1.2), (0.55, -3.5, 1.0), 70)}/>
    <camera name="aisle_cam" {look((0.55, 0.0, 0.9), (0.55, 3.0, 0.9), 75)}/>
    <!-- ceiling work lights -->
    <light name="lamp_a" pos="{vec(AX, -1.4, AZ + 1.35)}" dir="0 0.25 -1" cutoff="80" exponent="1" diffuse="0.50 0.50 0.53" castshadow="true"/>
    <light name="lamp_b" pos="{vec(AX, 0.4, AZ + 1.35)}" dir="0 0 -1" cutoff="80" exponent="1" diffuse="0.55 0.55 0.57" castshadow="true"/>
    <light name="lamp_c" pos="{vec(AX, 2.2, AZ + 1.35)}" dir="0 -0.2 -1" cutoff="80" exponent="1" diffuse="0.40 0.40 0.44" castshadow="false"/>
  </worldbody>
</mujoco>
"""
    (OUT / "mod_cameras.xml").write_text(xml)


def make_extra():
    xml = """<mujoco model="bas_extra_objects">
  <!-- Optional extra objects: sample canister and a small blue box on the bench (resting on the velcro surface). -->
  <worldbody>
    <body name="sample_canister" pos="0.30 0.33 0.415">
      <joint name="canister_free" type="free" damping="0.01"/>
      <geom name="canister_geom" type="cylinder" size="0.022 0.04" mass="0.06" rgba="0.15 0.55 0.65 1" friction="1.5 0.02 0.002" margin="0.003" gap="0.003"/>
    </body>
    <body name="tool_box" pos="0.82 -0.33 0.39">
      <joint name="toolbox_free" type="free" damping="0.01"/>
      <geom name="toolbox_geom" type="box" size="0.03 0.02 0.015" mass="0.05" rgba="0.2 0.3 0.8 1" friction="1.5 0.02 0.002" margin="0.003" gap="0.003"/>
    </body>
  </worldbody>
</mujoco>
"""
    (OUT / "mod_extra_objects.xml").write_text(xml)


def write_main(name, extra):
    extra_inc = '\n  <include file="mod_extra_objects.xml"/>' if extra else ""
    qpos = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04  0.45 0 0.40 1 0 0 0"
    if extra:
        qpos += "  0.30 0.33 0.415 1 0 0 0  0.82 -0.33 0.39 1 0 0 0"
    xml = f"""<mujoco model="bas_lab">
  <!-- Bharatiya Antariksh Station (BAS) science & robotics lab -- microgravity Panda workcell (generated by build_bas_lab.py).
       Add arms/sensors/tasks by adding more <include> lines (or <attach> for extra arms). -->
  <compiler angle="radian" meshdir="../mujoco_menagerie/franka_emika_panda/assets" texturedir="textures" autolimits="true"/>
  <option integrator="implicitfast" gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.22 0.23 0.26" diffuse="0.18 0.18 0.2" specular="0.05 0.05 0.05"/>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="4096"/>
  </visual>

  <include file="mod_panda_arm.xml"/>
  <include file="mod_bas_module.xml"/>
  <include file="mod_workstation.xml"/>{extra_inc}
  <include file="mod_cameras.xml"/>

  <keyframe>
    <!-- home pose, cube on the velcro bench, gripper open, velcro on (last ctrl) -->
    <key name="home" qpos="{qpos}" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255 1" mpos="0.60 0 0.3765"/>
  </keyframe>
</mujoco>
"""
    (OUT / name).write_text(xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_objects", action="store_true", help="also write bas_lab_extra_objects.xml (cube + canister + box)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    make_textures()
    write_panda_module()
    n = make_module(rng)
    make_workstation(rng, False)
    make_cameras()
    write_main("bas_lab.xml", False)
    if args.extra_objects:
        make_extra()
        write_main("bas_lab_extra_objects.xml", True)
    print(f"wrote {OUT}  ({n} racks)")


if __name__ == "__main__":
    main()
