"""
view_station.py -- open the space-station scene in the interactive MuJoCo window (look around, nothing is recorded).

    .\\.venv\\Scripts\\python.exe view_station.py            # station scene
    .\\.venv\\Scripts\\python.exe view_station.py --plain    # the plain marker scene, for comparison

Mouse: left-drag = rotate, right-drag = pan, scroll = zoom, double-click a body = select it.
Keys: space = pause/run physics, Tab = hide/show side panels, Backspace = reset, F1 = help, Esc/close = quit.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE_DIR = Path(__file__).resolve().parent / "mujoco_menagerie" / "franka_emika_panda"

ap = argparse.ArgumentParser()
ap.add_argument("--plain", action="store_true", help="show panda_pick_place_marker.xml instead of the station")
ap.add_argument("--bas", action="store_true", help="show the microgravity BAS lab (bas_lab/bas_lab.xml)")
ap.add_argument("--extra", action="store_true", help="with --bas: the 3-object variant")
args = ap.parse_args()

xml = SCENE_DIR / ("panda_pick_place_marker.xml" if args.plain else "panda_station_marker.xml")
if args.bas:
    xml = Path(__file__).resolve().parent / "bas_lab" / ("bas_lab_extra_objects.xml" if args.extra else "bas_lab.xml")
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)        # robot in its normal "home" pose
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    # start with the same view the policy camera uses, then you can move freely
    v.cam.lookat[:] = [0.55, 0.0, 0.43]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = (2.2 if not args.bas else 2.0), 135, -25
    print("Viewing:", xml.name, "- close the window to exit.")
    while v.is_running():
        mujoco.mj_step(model, data)
        data.ctrl[:] = model.key_ctrl[0]           # hold the home pose
        v.sync()
