"""
Prototype adapter: Panda joint-space demonstrations -> OpenVLA-style 7-D end-effector actions.

Pretrained OpenVLA predicts 7 numbers per step:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
(world-frame end-effector deltas in metres / radians, plus absolute gripper openness).

Our recorded data (episode_*.npz) stores 8-D absolute joint targets
    [q1..q7 (rad), gripper openness 0..1].

This module (nothing here steps the simulation or touches the originals):
  * encode:  (state, next_state) joint vectors -> 7-D EE delta action (forward kinematics);
             default label = achieved motion FK(s[t+1]) - FK(s[t]); the older commanded-target
             variant (FK(a[t]) - FK(s[t])) is kept as mode="commanded"
  * decode:  7-D EE delta action + state    -> 8-D joint target       (via damped least-squares IK)
  * stats:   OpenVLA-style q01/q99/mean/std normalization statistics
  * main():  converts every demonstrations/episode_*.npz, prints statistics and a
             round-trip check (encode -> decode -> compare against recorded joint targets).

The end-effector is the MuJoCo body "hand", matching record_demos.py / pick_place.py.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR / "mujoco_menagerie" / "franka_emika_panda" / "panda_pick_place.xml"
)
DEMO_DIR = PROJECT_DIR / "demonstrations"
OUT_DIR = PROJECT_DIR / "openvla_adapter_out"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
EE_BODY = "hand"


# --------------------------------------------------------------------------- rotations
def rot_to_euler_xyz(R):
    """Rotation matrix -> (roll, pitch, yaw), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def euler_xyz_to_rot(e):
    r, p, y = e
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_to_rotvec(R):
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


# --------------------------------------------------------------------------- kinematics
class PandaKinematics:
    """Forward kinematics and damped-least-squares IK for the Panda 'hand' body."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        if self.hand_id < 0:
            raise ValueError(f"Body '{EE_BODY}' not found in the scene.")

        jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS
        ]
        if min(jids) < 0:
            raise ValueError("Missing Panda arm joints in the scene.")
        self.qpos_idx = np.array([self.model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dof_idx = np.array([self.model.jnt_dofadr[j] for j in jids], dtype=int)
        self.q_lo = np.array([self.model.jnt_range[j][0] for j in jids])
        self.q_hi = np.array([self.model.jnt_range[j][1] for j in jids])

    def fk(self, q):
        """Joint vector (7,) -> (position (3,), rotation matrix (3,3)) of the hand."""
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jacBody
        return (
            self.data.xpos[self.hand_id].copy(),
            self.data.xmat[self.hand_id].reshape(3, 3).copy(),
        )

    def ik(self, q_init, target_pos, target_rot, iters=100, damping=1e-2, tol=1e-5):
        """Damped least-squares IK from q_init. Returns (q, position_error_m)."""
        q = np.array(q_init, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err_norm = np.inf
        for _ in range(iters):
            pos, rot = self.fk(q)
            err = np.concatenate([target_pos - pos, rot_to_rotvec(target_rot @ rot.T)])
            err_norm = np.linalg.norm(err)
            if err_norm < tol:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            J = np.vstack([jacp[:, self.dof_idx], jacr[:, self.dof_idx]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        pos, _ = self.fk(q)
        return q, float(np.linalg.norm(target_pos - pos))


# --------------------------------------------------------------------------- encode / decode
def encode_action(kin, state, action):
    """
    (8-D state, 8-D joint-target action) -> 7-D [dx,dy,dz, droll,dpitch,dyaw, gripper].
    Deltas are the world-frame hand displacement from FK(state) to FK(action target).
    """
    p_s, R_s = kin.fk(state[:7])
    p_a, R_a = kin.fk(action[:7])
    dpos = p_a - p_s
    drot = rot_to_euler_xyz(R_a @ R_s.T)  # world-frame relative rotation
    return np.concatenate([dpos, drot, [action[7]]]).astype(np.float32)


def encode_action_next_state(kin, state, next_state, action):
    """
    Same 7-D format, but the delta is the *achieved* hand displacement over one step:
    FK(next measured state) - FK(current measured state). The gripper entry is the commanded
    openness (absolute). Preferred label: it is what a delta controller must reproduce, and it has
    no servo-lag offset (the commanded-target delta over-states motion when executed as a held
    target; see closed_loop_panda.py).
    """
    p0, R0 = kin.fk(state[:7])
    p1, R1 = kin.fk(next_state[:7])
    return np.concatenate([p1 - p0, rot_to_euler_xyz(R1 @ R0.T), [action[7]]]).astype(np.float32)


def decode_action(kin, state, action7):
    """
    (8-D state, 7-D EE delta action) -> 8-D joint-target action for PandaActionInterface.
    Returns (action8, position_residual_m).
    """
    action7 = np.asarray(action7, dtype=np.float64)
    p_s, R_s = kin.fk(state[:7])
    target_pos = p_s + action7[:3]
    target_rot = euler_xyz_to_rot(action7[3:6]) @ R_s
    q, resid = kin.ik(state[:7], target_pos, target_rot)
    grip = float(np.clip(action7[6], 0.0, 1.0))
    return np.concatenate([q, [grip]]).astype(np.float32), resid


# --------------------------------------------------------------------------- stats
def compute_stats(actions7):
    """OpenVLA-style norm stats. Gripper dimension is left un-normalized (mask False)."""
    a = np.asarray(actions7, dtype=np.float64)
    return {
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).tolist(),
        "mask": [True] * 6 + [False],
    }


# --------------------------------------------------------------------------- main
def convert_episode(kin, path, mode="next_state"):
    """mode: 'next_state' (achieved motion, default) or 'commanded' (FK(target) - FK(state))."""
    d = np.load(path, allow_pickle=True)
    states = d["states"].astype(np.float64)
    actions = d["actions"].astype(np.float64)
    if mode == "commanded":
        a7 = np.stack([encode_action(kin, s, a) for s, a in zip(states, actions)])
    elif mode == "next_state":
        nxt = np.vstack([states[1:], states[-1:]])  # last frame: no further motion
        a7 = np.stack([encode_action_next_state(kin, s, n, a) for s, n, a in zip(states, nxt, actions)])
    else:
        raise ValueError(mode)
    return {
        "images": d["images"],
        "instruction": str(d["instruction"]),
        "states": states,
        "actions_joint": actions,
        "actions7": a7,
    }


def main():
    paths = sorted(DEMO_DIR.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episodes in {DEMO_DIR}")
    kin = PandaKinematics()
    OUT_DIR.mkdir(exist_ok=True)

    all_a7 = []
    print(f"Converting {len(paths)} episodes (read-only on {DEMO_DIR.name}/)\n")
    for path in paths:
        ep = convert_episode(kin, path)
        a7 = ep["actions7"]
        all_a7.append(a7)

        # Round trip: decode each 7-D action from the recorded state, compare with the
        # recorded joint target (position error of the hand and joint-space error).
        pos_err, joint_err, ik_resid = [], [], []
        nxt = np.vstack([ep["states"][1:], ep["states"][-1:]])
        for s, n, a_ee in zip(ep["states"], nxt, a7):
            a8_dec, resid = decode_action(kin, s, a_ee)
            p_true, _ = kin.fk(n[:7])          # achieved hand pose at the next frame
            p_dec, _ = kin.fk(a8_dec[:7])
            pos_err.append(np.linalg.norm(p_true - p_dec))
            joint_err.append(np.max(np.abs(n[:7] - a8_dec[:7])))
            ik_resid.append(resid)

        print(f"{path.name}: {len(a7)} frames | \"{ep['instruction']}\"")
        print(f"  |dpos| per step   mean {np.linalg.norm(a7[:, :3], axis=1).mean() * 1000:.2f} mm,"
              f" max {np.linalg.norm(a7[:, :3], axis=1).max() * 1000:.2f} mm")
        print(f"  |drot| per step   max {np.abs(a7[:, 3:6]).max():.4f} rad")
        print(f"  gripper range     {a7[:, 6].min():.2f} .. {a7[:, 6].max():.2f}")
        print(f"  round-trip hand position error   max {max(pos_err) * 1000:.4f} mm")
        print(f"  round-trip max joint difference  max {max(joint_err):.4f} rad"
              f" (redundant 7-DoF arm: differs in null-space, not an error by itself)")
        print(f"  IK residual                      max {max(ik_resid) * 1000:.4f} mm\n")

    all_a7 = np.concatenate(all_a7)
    stats = compute_stats(all_a7)
    out = OUT_DIR / "action_stats_pilot.json"
    out.write_text(json.dumps({"panda_ee_delta": {"action": stats}}, indent=2))

    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    print("Pooled 7-D action statistics (all episodes)")
    print(f"  {'dim':7s} {'q01':>9s} {'q99':>9s} {'mean':>9s} {'std':>9s}")
    for i, n in enumerate(names):
        print(f"  {n:7s} {stats['q01'][i]:9.4f} {stats['q99'][i]:9.4f}"
              f" {stats['mean'][i]:9.4f} {stats['std'][i]:9.4f}")
    print(f"\nWrote {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
