"""
closed_loop_panda.py -- run a policy in the Panda MuJoCo scene, closed loop at 10 Hz.

Each policy step:
    render 320x240 image -> resize 224x224 -> policy -> 7-D EE delta [dx,dy,dz,droll,dpitch,dyaw,grip]
    -> tracking controller (IK + scratch-sim refinement) -> 8-D joint target -> PandaActionInterface
    -> 50 physics steps

Policies
    --policy replay   feed the *recorded* 7-D actions of the chosen episode, frame by frame, through the
                      same decode/clamp/step path. No model: validates the simulation + control path.
    --policy server   query policy_server.py through http://127.0.0.1:<port>.

Scenarios are taken (read-only) from demonstrations/episode_XXX.npz: initial cube xy, target xy,
instruction. Results go to closed_loop_results/<tag>/ ; demonstrations/ is never modified.

Examples
    python closed_loop_panda.py --policy replay --episodes 0 1 2 3
    python closed_loop_panda.py --policy server --episodes 0 1 2 3 --tag lora_dryrun
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import openvla_adapter as A
import panda_openvla_dataset as P
from panda_action_interface import PandaActionInterface

PROJECT_DIR = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_DIR / "demonstrations"
DATA_DIR = PROJECT_DIR / "panda_openvla_data"
RESULT_DIR = PROJECT_DIR / "closed_loop_results"

IMAGE_W, IMAGE_H = 320, 240
STEPS_PER_POLICY = 50          # 0.002 s timestep -> 10 Hz, same as recording
GRIPPER_MAX_CTRL = 255.0
FINGER_OPENING_MAX_M = 0.04
TABLE_TOP_Z = 0.375
CUBE_HALF = 0.025
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF

# Safety limits applied to every model-generated command (before it reaches the actuators).
MAX_JOINT_STEP = 0.15          # rad per policy step, per joint (demo max is far below this)
MAX_HAND_STEP = 0.03           # m per policy step (demo max ~0.021)
SUCCESS_XY_M = 0.04
SUCCESS_HOLD_STEPS = 10
# Success rule v2 (D69, 2026-11-04): a cube that is merely lowered while still held no longer counts.
#   placed   = cube within SUCCESS_XY_M of the marker and on the surface, gripper reading > 0.5 (the old rule; a held
#              5 cm cube reads ~0.62-0.69, so this only says "lowered onto the marker")      -> reported as "placed"
#   released = placed geometry + gripper reading > RELEASE_READING + no contact cube/hand/fingers   -> "success"
#   upright  = released and the cube tilt < UPRIGHT_TILT_DEG                                           -> "upright"
RELEASE_READING = 0.8
UPRIGHT_TILT_DEG = 15.0
TRACK_PRIOR_GAIN = 2.0         # measured: a held target delivers ~half its offset within 0.1 s
TRACK_ITERS = 3                # fixed-point refinements on a scratch copy of the sim


MARKER_SCENE = A.MODEL_PATH.with_name("panda_pick_place_marker.xml")
BAS_SCENE = Path(__file__).resolve().parent / "bas_lab" / "bas_lab.xml"   # microgravity station lab (same coordinates)
SIDE_CAMERA = ((0.50, 0.0, 0.46), 0.75, 90, -15)   # lookat, distance, azimuth, elevation (D67)
MARKER_Z = 0.3765  # centre of the 1 mm thick marker resting on the table top


class PandaEnv:
    def __init__(self, scene_xml=None, camera="default"):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml or A.MODEL_PATH))
        self.has_marker = self.model.nmocap > 0
        self.step_hook = None  # optional callable run after every physics step (e.g. viewer sync)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.home_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        assert min(self.home_id, self.hand_id, self.cube_id) >= 0, "scene is missing home/hand/pick_cube"
        self.finger_idx = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ("finger_joint1", "finger_joint2")], dtype=int)
        first = int(m.body_jntadr[self.cube_id])
        self.cube_qadr = next(
            int(m.jnt_qposadr[j]) for j in range(first, first + int(m.body_jntnum[self.cube_id]))
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)

        self.interface = PandaActionInterface(m)
        self.arm_idx = self.interface.arm_qpos_indices
        self.renderer = mujoco.Renderer(m, height=IMAGE_H, width=IMAGE_W)
        cam = mujoco.MjvCamera()                    # identical to record_demos.py
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array([0.55, 0.0, 0.43])
        cam.distance, cam.azimuth, cam.elevation = 1.65, 135, -25
        if camera == "side":                         # closer side camera (D67/D72): finger gap and raised cube visible
            cam.lookat[:] = np.array(SIDE_CAMERA[0])
            cam.distance, cam.azimuth, cam.elevation = SIDE_CAMERA[1:]
        self.camera = cam
        self.kin = A.PandaKinematics()              # separate MjData: never touches the sim state

    def reset(self, cube_xy, target_xy=None):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home_id)
        d.qpos[self.cube_qadr:self.cube_qadr + 7] = [cube_xy[0], cube_xy[1], CUBE_REST_Z, 1, 0, 0, 0]
        if self.has_marker and target_xy is not None:  # visible target (marker scene only)
            d.mocap_pos[0] = [target_xy[0], target_xy[1], MARKER_Z]
        mujoco.mj_forward(m, d)
        d.ctrl[:7] = d.qpos[self.arm_idx]
        d.ctrl[7] = GRIPPER_MAX_CTRL

    def cube_touches_gripper(self):
        """True if the cube is in contact (incl. the 3 mm margin of the lab scene) with the hand or a finger."""
        if not hasattr(self, "_gripper_bodies"):
            m = self.model
            self._gripper_bodies = {self.hand_id} | {
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")}
        d, m = self.data, self.model
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            if (b1 == self.cube_id and b2 in self._gripper_bodies) or (b2 == self.cube_id and b1 in self._gripper_bodies):
                return True
        return False

    def cube_tilt_deg(self):
        """Angle between the nearest cube axis and world z: 0 = lying flat on a face, ~54.7 = balanced on a corner."""
        R = self.data.xmat[self.cube_id].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), 0.0, 1.0))))

    def image(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def state(self):
        d = self.data
        opening = float(np.mean(d.qpos[self.finger_idx]))
        grip = float(np.clip(opening / FINGER_OPENING_MAX_M, 0.0, 1.0))
        return np.concatenate([d.qpos[self.arm_idx], [grip]])

    def cube(self):
        return self.data.xpos[self.cube_id].copy()

    def hand(self):
        return self.data.xpos[self.hand_id].copy()

    def apply_joint_target(self, target8):
        """Clamp the commanded step, write ctrl, advance physics. Returns (clamped target, was_clamped)."""
        d = self.data
        q = d.qpos[self.arm_idx]
        arm = np.clip(target8[:7], q - MAX_JOINT_STEP, q + MAX_JOINT_STEP)
        clamped = bool(np.any(np.abs(arm - target8[:7]) > 1e-9))
        self.interface.apply_action(d, np.concatenate([arm, [target8[7]]]))
        for _ in range(STEPS_PER_POLICY):
            mujoco.mj_step(self.model, d)
        return np.concatenate([arm, [target8[7]]]), clamped

    def _rollout_displacement(self, target8):
        """Apply target8 for one policy step on a scratch copy of the sim; return the achieved
        hand displacement (position, euler rotation) relative to the current hand pose."""
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        self.interface.apply_action(scratch, target8)
        for _ in range(STEPS_PER_POLICY):
            mujoco.mj_step(self.model, scratch)
        return scratch.xpos[self.hand_id].copy(), scratch.xmat[self.hand_id].reshape(3, 3).copy()

    def execute_delta(self, a7, iters=TRACK_ITERS):
        """
        Low-level end-effector-delta controller. Finds the joint target whose 50-step response moves
        the hand by the requested displacement a7[:6] (measured, not commanded), then applies it.

        The position servos only deliver part of a held target's offset within 0.1 s, so the
        commanded offset is refined by fixed-point iteration on a scratch copy of the simulator
        (c <- c + (desired - achieved)), starting from the empirically measured gain of ~2.
        Returns (applied 8-D joint target, was_clamped, residual_position_m).
        """
        state = self.state()
        p_s, R_s = self.kin.fk(state[:7])
        d_pos, d_rot, grip = a7[:3], a7[3:6], float(a7[6])
        c_pos, c_rot = TRACK_PRIOR_GAIN * d_pos, TRACK_PRIOR_GAIN * d_rot
        q = self.data.qpos[self.arm_idx]
        resid = 0.0
        for k in range(iters + 1):
            target8, _ = A.decode_action(self.kin, state, np.concatenate([c_pos, c_rot, [grip]]))
            target8 = target8.astype(np.float64)
            target8[:7] = np.clip(target8[:7], q - MAX_JOINT_STEP, q + MAX_JOINT_STEP)
            p1, R1 = self._rollout_displacement(target8)
            ach_pos, ach_rot = p1 - p_s, A.rot_to_euler_xyz(R1 @ R_s.T)
            resid = float(np.linalg.norm(d_pos - ach_pos))
            if k == iters or resid < 2e-4:
                break
            c_pos = c_pos + (d_pos - ach_pos)
            c_rot = c_rot + (d_rot - ach_rot)
        clamped = bool(np.any(np.abs(target8[:7] - A.decode_action(
            self.kin, state, np.concatenate([c_pos, c_rot, [grip]]))[0][:7]) > 1e-9))
        self.interface.apply_action(self.data, target8)
        for _ in range(STEPS_PER_POLICY):
            mujoco.mj_step(self.model, self.data)
            if self.step_hook is not None:
                self.step_hook()
        return target8, clamped, resid

    def close(self):
        self.renderer.close()


# --------------------------------------------------------------------------- policies
class ReplayPolicy:
    """Feeds back the recorded 7-D actions of one episode (validation of the control path only)."""

    def __init__(self, episode):
        d = np.load(DATA_DIR / "data.npz")
        idx = episode
        split_path = DATA_DIR / "split.json"
        if split_path.exists():  # exports may be renumbered: map file number -> export index
            eps = json.loads(split_path.read_text())["episodes"]
            idx = next(e["episode"] for e in eps if e["file"] == f"episode_{episode:03d}.npz")
        sel = d["episode_index"] == idx
        self.actions = d["actions7"][sel]
        self.t = 0

    def __call__(self, image_224, instruction):
        a = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return a.astype(np.float64), 0.0


class ServerPolicy:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            self.info = json.loads(r.read())

    def __call__(self, image_224, instruction):
        body = json.dumps({
            "instruction": instruction,
            "image_b64": base64.b64encode(np.ascontiguousarray(image_224).tobytes()).decode(),
            "shape": list(image_224.shape),
        }).encode()
        out, last_err = None, None
        for attempt in range(4):  # a stalled link must not abort a whole evaluation run
            try:
                req = urllib.request.Request(self.url + "/act", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    out = json.loads(r.read())
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                print(f"    [policy request failed ({e!r}); retry {attempt + 1}/3]", flush=True)
                time.sleep(5)
        if out is None:
            raise RuntimeError(f"policy server unreachable after 4 attempts: {last_err!r}")
        if "error" in out:
            raise RuntimeError(f"policy server error: {out['error']}")
        return np.asarray(out["action"], dtype=np.float64), float(out["latency_s"])


# --------------------------------------------------------------------------- episode loop
# Held-out layouts 380-391 (cube start xy, marker xy), so they work without the demonstration files.
HELD_OUT_LAYOUTS = {
    380: ((0.413, -0.011), (0.617, -0.086)), 381: ((0.420, 0.059), (0.591, -0.134)),
    382: ((0.464, -0.096), (0.606, 0.132)), 383: ((0.452, 0.038), (0.577, -0.063)),
    384: ((0.448, -0.013), (0.608, -0.106)), 385: ((0.426, 0.064), (0.576, -0.157)),
    386: ((0.409, 0.057), (0.578, 0.056)), 387: ((0.417, -0.002), (0.643, 0.161)),
    388: ((0.458, 0.091), (0.589, -0.015)), 389: ((0.410, -0.077), (0.572, -0.180)),
    390: ((0.458, 0.069), (0.608, 0.167)), 391: ((0.445, 0.056), (0.619, 0.033)),
}
DEFAULT_INSTRUCTION = "Pick up the red cube and place it on the yellow marker."


def load_scenario(episode):
    path = DEMO_DIR / f"episode_{episode:03d}.npz"
    if not path.exists() and episode in HELD_OUT_LAYOUTS:
        s, t = HELD_OUT_LAYOUTS[episode]
        return DEFAULT_INSTRUCTION, np.asarray(s, float), np.asarray(t, float)
    d = np.load(path, allow_pickle=True)
    cube = d["cube_positions"]
    start_xy = d["start_xy"] if "start_xy" in d.files else cube[0][:2]
    target_xy = d["target_xy"] if "target_xy" in d.files else cube[-1][:2]
    return str(d["instruction"]), np.asarray(start_xy, float), np.asarray(target_xy, float)


TRAINED_INSTRUCTION = "Pick up the red cube and place it on the yellow marker."
# Same sampling ranges as the demo recorders / the expert (the distribution the model was trained on).
_START_X, _START_Y = (0.40, 0.49), (-0.10, 0.10)
_MARK_X, _MARK_Y = (0.57, 0.66), (-0.20, 0.20)


def random_scene(rng):
    """Fresh random cube start and marker position, at least 13 cm apart."""
    s = (float(rng.uniform(*_START_X)), float(rng.uniform(*_START_Y)))
    for _ in range(200):
        t = (float(rng.uniform(*_MARK_X)), float(rng.uniform(*_MARK_Y)))
        if np.hypot(s[0] - t[0], s[1] - t[1]) >= 0.13:
            return s, t
    return s, t


# Optional gripper latch (--gripper_latch, off by default; results with it are reported separately).
# The single-frame policy chatters between "open, down" and "close, up" a little above the grasp height;
# the latch does what the scripted expert's CLOSE phase does with the finger sensor: once the policy asks
# for "close" low over the cube, stand still and close until the fingers touch the cube, then keep the
# gripper closed while the policy lifts.
LATCH_Z = 0.53          # only below this hand height
LATCH_CONTACT = 0.75    # gripper reading below this = fingers are on the cube
LATCH_MAX_CLOSE = 12    # steps standing still while closing
LATCH_HOLD = 25         # steps the gripper stays forced closed after contact (about 10 cm of lift)
LATCH_COOL = 20         # no re-trigger after a closed-on-nothing abort


def run_episode(env, policy, instruction, start_xy, target_xy, max_steps, save_frames, verbose=True, latch=False,
                legacy=False):
    """legacy=True reproduces the old rule (the run stops as soon as the cube is lowered onto the marker)."""
    env.reset(start_xy, target_xy)
    instr = P.format_instruction(instruction)
    log = {"hand": [], "cube": [], "grip_obs": [], "action7": [], "latency": [], "clamped": 0,
           "policy_grip": [], "latch": [], "touch": [], "tilt": []}
    frames = []
    held_placed = held_rel = 0
    placed_step = released_step = None
    status = "max_steps"
    mode, mode_t, cool = "off", 0, 0     # latch state: off | closing | holding

    for t in range(max_steps):
        img = env.image()
        if save_frames:
            frames.append(img)
        state = env.state()
        a7, lat = policy(P.resize_image(img), instr)
        log["policy_grip"].append(float(a7[6]))
        if latch:
            a7 = np.array(a7, dtype=np.float64, copy=True)
            cool = max(cool - 1, 0)
            if mode == "off" and cool == 0 and a7[6] < 0.5 and env.hand()[2] < LATCH_Z and state[7] > LATCH_CONTACT:
                mode, mode_t = "closing", 0
            if mode == "closing":
                a7[:6], a7[6] = 0.0, 0.0
                mode_t += 1
                if state[7] < 0.3:                       # closed on nothing: give up
                    mode, cool = "off", LATCH_COOL
                elif state[7] < LATCH_CONTACT or mode_t >= LATCH_MAX_CLOSE:
                    mode, mode_t = "holding", 0
            elif mode == "holding":
                a7[6] = 0.0
                mode_t += 1
                if mode_t >= LATCH_HOLD:
                    mode = "off"
        log["latch"].append(mode != "off")

        # Safety: bound the end-effector step before IK.
        n = np.linalg.norm(a7[:3])
        if n > MAX_HAND_STEP:
            a7 = a7.copy(); a7[:3] *= MAX_HAND_STEP / n
            log["clamped"] += 1
        if not np.all(np.isfinite(a7)):
            status = "nonfinite_action"; break

        _, jclamp, resid = env.execute_delta(a7)
        log["clamped"] += int(jclamp)
        log.setdefault("track_resid", []).append(resid)

        cube = env.cube()
        log["hand"].append(env.hand()); log["cube"].append(cube); log["grip_obs"].append(state[7])
        log["action7"].append(a7); log["latency"].append(lat)

        if not np.all(np.isfinite(cube)) or cube[2] < TABLE_TOP_Z - 0.10:
            status = "cube_fell"; break
        xy_err = np.linalg.norm(cube[:2] - target_xy)
        grip_now = env.state()[7]
        touching, tilt = env.cube_touches_gripper(), env.cube_tilt_deg()
        log["touch"].append(touching); log["tilt"].append(tilt)
        on_marker = xy_err < SUCCESS_XY_M and abs(cube[2] - CUBE_REST_Z) < 0.02
        held_placed = held_placed + 1 if (on_marker and grip_now > 0.5) else 0
        held_rel = held_rel + 1 if (on_marker and grip_now > RELEASE_READING and not touching) else 0
        if held_placed >= SUCCESS_HOLD_STEPS and placed_step is None:
            placed_step = t
        if legacy and placed_step is not None:
            status = "success"; break
        if held_rel >= SUCCESS_HOLD_STEPS:
            status, released_step = "success", t; break
        if verbose and t % 25 == 0:
            print(f"    t={t:3d} hand={np.round(env.hand(), 3)} cube={np.round(cube, 3)} "
                  f"grip={state[7]:.2f} xy_err={xy_err * 100:.1f}cm", flush=True)

    cube_arr = np.asarray(log["cube"])
    final_xy_err = float(np.linalg.norm(cube_arr[-1][:2] - target_xy)) if len(cube_arr) else float("nan")
    res = {
        "status": status,
        "success": status == "success",
        "rule": "legacy_placed" if legacy else "released_v2",
        "placed": placed_step is not None,
        "placed_step": placed_step,
        "released_step": released_step,
        "upright": bool(status == "success" and not legacy and log["tilt"] and log["tilt"][-1] < UPRIGHT_TILT_DEG),
        "final_tilt_deg": float(log["tilt"][-1]) if log["tilt"] else None,
        "steps": len(cube_arr),
        "final_xy_error_m": final_xy_err,
        "max_cube_lift_m": float(cube_arr[:, 2].max() - CUBE_REST_Z) if len(cube_arr) else 0.0,
        "clamped_steps": int(log["clamped"]),
        "mean_latency_s": float(np.mean(log["latency"])) if log["latency"] else 0.0,
        "mean_tracking_residual_mm": float(np.mean(log.get("track_resid", [0.0])) * 1000),
    }
    return res, log, frames


def save_gif(frames, path, every=2):
    if not frames:
        return
    ims = [Image.fromarray(f) for f in frames[::every]]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=100 * every, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["replay", "server"], required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_steps", type=int, default=320)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="OpenVLA export used by --policy replay (default: panda_openvla_data)")
    ap.add_argument("--no_gif", action="store_true")
    ap.add_argument("--random", type=int, default=0,
                    help="evaluate on N fresh random scenes instead of --episodes (server policy only); "
                         "scenes get ids 1000+i, their start/target are stored in summary.json")
    ap.add_argument("--seed", type=int, default=777, help="seed for --random")
    ap.add_argument("--scene", choices=["original", "marker", "bas"], default="original",
                    help="marker = scene with a visible yellow target marker at each scenario's target; "
                         "bas = microgravity space-station lab (bas_lab/bas_lab.xml, same marker coordinates)")
    ap.add_argument("--camera", choices=["default", "side"], default="default",
                    help="side = closer side camera (D67); the model must have been trained on it")
    ap.add_argument("--legacy_success", action="store_true",
                    help="old rule: stop and count success as soon as the cube is lowered onto the marker, "
                         "even if it is still held (see RELEASE_READING); only for comparing with old results")
    ap.add_argument("--gripper_latch", action="store_true",
                    help="hold the gripper closed once the policy asks to close low over the cube (see LATCH_*); "
                         "not part of the standard protocol, report such results separately")
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    tag = args.tag or args.policy
    out_dir = RESULT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.random and args.policy != "server":
        raise SystemExit("--random needs --policy server")
    scenarios = []
    if args.random:
        rng = np.random.default_rng(args.seed)
        for i in range(args.random):
            s_, t_ = random_scene(rng)
            scenarios.append((1000 + i, TRAINED_INSTRUCTION, np.asarray(s_, float), np.asarray(t_, float)))
    else:
        for ep in args.episodes:
            instr_, s_, t_ = load_scenario(ep)
            scenarios.append((ep, instr_, s_, t_))

    env = PandaEnv({"marker": MARKER_SCENE, "bas": BAS_SCENE}.get(args.scene), camera=args.camera)
    server = ServerPolicy(args.port) if args.policy == "server" else None
    if server:
        print("policy server:", server.info, flush=True)

    results = {}
    try:
        for ep, instr, s_xy, t_xy in scenarios:
            print(f"\n[episode {ep}] \"{instr}\"\n  start_xy={np.round(s_xy, 3)} target_xy={np.round(t_xy, 3)}", flush=True)
            policy = ReplayPolicy(ep) if args.policy == "replay" else server
            t0 = time.time()
            res, log, frames = run_episode(env, policy, instr, s_xy, t_xy, args.max_steps, save_frames=not args.no_gif,
                                           latch=args.gripper_latch, legacy=args.legacy_success)
            res["wall_s"] = round(time.time() - t0, 1)
            res["start_xy"] = [float(v) for v in s_xy]
            res["target_xy"] = [float(v) for v in t_xy]
            results[f"episode_{ep:03d}"] = res
            np.savez_compressed(out_dir / f"episode_{ep:03d}_log.npz",
                                **{k: np.asarray(v) for k, v in log.items() if k != "clamped"})
            if not args.no_gif:
                save_gif(frames, out_dir / f"episode_{ep:03d}.gif")
            print("  ->", json.dumps(res), flush=True)
    finally:
        env.close()

    n_ok = sum(r["success"] for r in results.values())
    n_placed = sum(r["placed"] for r in results.values())
    n_up = sum(r["upright"] for r in results.values())
    n = max(len(results), 1)
    summary = {"policy": args.policy, "rule": "legacy_placed" if args.legacy_success else "released_v2",
               "episodes": results, "success_rate": n_ok / n, "placed_rate": n_placed / n, "upright_rate": n_up / n}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.legacy_success:
        print(f"\nSUCCESS {n_ok}/{len(results)} (legacy rule: cube lowered onto the marker)   "
              f"results in {out_dir.relative_to(PROJECT_DIR)}/")
    else:
        print(f"\nSUCCESS {n_ok}/{len(results)} (released)  |  placed (old rule) {n_placed}/{len(results)}  |  "
              f"upright {n_up}/{len(results)}   results in {out_dir.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
