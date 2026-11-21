"""
multitask_expert.py -- closed-loop scripted experts for station tasks 2-5 (task 1 keeps expert_policy.ExpertPolicy).

Each task is a list of stages. Every step the expert reads the true state sent by Unreal (object / target / handle
positions in MuJoCo metres, drawer opening, valve and lid angles, held and stowed flags), moves the current stage's
goal towards it and outputs the same 7-D action as the model: [dx, dy, dz, droll, dpitch, dyaw, gripper] (world-frame
hand deltas; gripper 1 = open, 0 = closed). Losing a held object or handle sends it back to that skill's first stage.

    2  stow tool:   open drawer (grip knob, pull +y), pick tool, place it in the drawer, push the drawer shut
    3  valve:       grip lever, turn the wrist 90 degrees, let go
    4  filter:      pick cartridge, carry over the scrubber slot, lower it in, let go
    5  container:   pick vial, drop it into the container, grip the lid tab, swing the lid shut, let go
"""

import numpy as np

import openvla_adapter as A
from expert_policy import move_toward

GRASP_BELOW_HAND = 0.094            # grasp centre between the finger pads, below the hand body (hand pointing down)
DRAWER_MAX_OPEN = 0.11
LID_HINGE = np.array([-0.05, 0.0, 0.06])   # relative to the container centre on the bench
BENCH_Z = 0.375
V_FREE, V_FINE, V_HANDLE = 0.005, 0.004, 0.004
TURN_RATE = 0.04                    # rad per step while turning the valve
SWING_STEP = np.radians(3.0)        # lid angle per step while closing

# DART noise per stage (0 = must stay clean: contact, gripper changes, turning)
NOISE = {"REACH": 1.0, "DESCEND": 0.4, "CLOSE": 0.0, "LIFT": 0.4, "TRANSFER": 1.0, "LOWER": 0.4, "RELEASE": 0.0,
         "RISE": 0.4, "H_REACH": 1.0, "H_DESCEND": 0.4, "GRIP": 0.0, "PULL": 0.3, "PUSH": 0.3, "TURN": 0.0,
         "SWING": 0.3, "LETGO": 0.0, "REOPEN": 0.0, "DONE": 0.0}
OPEN_STAGES = {"REACH", "DESCEND", "RELEASE", "RISE", "H_REACH", "H_DESCEND", "LETGO", "REOPEN", "DONE"}


def above(p, dz=0.0):
    """Hand position that puts the grasp centre at p (plus dz)."""
    return np.asarray(p, float) + [0.0, 0.0, GRASP_BELOW_HAND + dz]


class TaskState:
    """Unreal's `state` object as arrays."""

    def __init__(self, raw):
        get = lambda k, d=0.0: float(raw.get(k, d))
        vec = lambda n: np.array([get(n + "_x"), get(n + "_y"), get(n + "_z")])
        self.obj, self.target = vec("object"), vec("target")
        self.handle = vec("handle") if "handle_x" in raw else None
        self.held_obj, self.held_handle = get("held_object") > 0.5, get("held_handle") > 0.5
        self.drawer, self.valve, self.lid = get("drawer_open"), get("valve_deg"), get("lid_deg", 90.0)
        self.tool_in, self.vial_in = get("tool_stowed") > 0.5, get("vial_stowed") > 0.5
        self.seated, self.latched = get("seated") > 0.5, get("latched") > 0.5


def task_success(task, s):
    """Task check on the final state (mirrors ABASWorkcellActor::IsTaskSuccess, without the gripper reading)."""
    return {2: s.tool_in and s.drawer <= 0.001, 3: abs(s.valve) >= 89.0, 4: s.seated,
            5: s.vial_in and s.latched}.get(task, False)


class MultiTaskExpert:
    def __init__(self, task, nominal_rot):
        self.task = task
        self.R_nom = np.asarray(nominal_rot, float)
        self.R_goal = self.R_nom.copy()
        self.carry_z = {2: 0.62, 3: 0.62, 4: 0.64, 5: 0.64}[task]   # task 5 passes beside the open lid
        self.stages = self._plan(task)
        self.i, self.steps_in_stage, self.recoveries = 0, 0, 0
        self.reopen_to = None           # set while recovering: open the gripper, then restart at this stage
        self.history = [(0, self.stages[0][0])]

    # ---- stage tables: (name, hold, retry) ; hold = what must stay gripped, retry = stage index on loss ----
    def _plan(self, task):
        pick = [("REACH", None, 0), ("DESCEND", None, 0), ("CLOSE", None, 0), ("LIFT", "obj", 0),
                ("TRANSFER", "obj", 0), ("LOWER", "obj", 0), ("RELEASE", None, 0), ("RISE", None, 0)]
        grip = lambda: [("H_REACH", None, 0), ("H_DESCEND", None, 0), ("GRIP", None, 0)]
        letgo = [("LETGO", None, 0), ("RISE", None, 0)]
        if task == 2:
            stages = grip() + [("PULL", "handle", 0)] + letgo + pick + grip() + [("PUSH", "handle", 0)] + letgo
        elif task == 3:
            stages = grip() + [("TURN", "handle", 0)] + letgo
        elif task == 4:
            stages = pick
        elif task == 5:
            stages = pick + grip() + [("SWING", "handle", 0)] + letgo
        else:
            raise ValueError(f"no multi-stage expert for task {task}")
        # each skill block retries from its own first stage
        out, block_start = [], 0
        for k, (name, hold, _) in enumerate(stages):
            if name in ("REACH", "H_REACH"):
                block_start = k
            out.append((name, hold, block_start))
        return out + [("DONE", None, len(out))]

    def _go(self, i, t):
        self.i, self.steps_in_stage = i, 0
        self.history.append((t, self.stages[i][0]))

    def _lid_goal(self, hand, s):
        """Next hand position on the lid tab's circle around the hinge, SWING_STEP closer to shut."""
        hinge = np.array([s.target[0], s.target[1], s.target[2] - 0.006]) + LID_HINGE
        g = hand - [0.0, 0.0, GRASP_BELOW_HAND] - hinge
        r, phi = np.hypot(g[0], g[2]), np.arctan2(g[2], g[0])
        phi_next = phi - min(SWING_STEP, np.radians(s.lid) + 0.01)
        return hinge + [r * np.cos(phi_next), g[1], r * np.sin(phi_next)] + [0.0, 0.0, GRASP_BELOW_HAND]

    def act(self, hand, R_hand, grip_obs, s, t):
        """hand: hand body xyz; R_hand: 3x3; grip_obs: opening 0..1; s: TaskState. Returns (a7, stage name)."""
        hand = np.asarray(hand, float)
        name, hold, retry = self.stages[self.i]

        # lost what we were holding -> open the gripper, then retry the skill from its first stage
        if self.reopen_to is None and ((hold == "obj" and not s.held_obj) or (hold == "handle" and not s.held_handle)):
            self.recoveries += 1
            self.reopen_to, self.steps_in_stage = retry, 0
            self.history.append((t, "REOPEN"))
        if self.reopen_to is not None:
            name = "REOPEN"

        goal, v = None, V_FREE
        handle = s.handle if s.handle is not None else s.obj
        carry = self.carry_z
        done = False
        if name == "REACH":
            goal = above(s.obj, 0.06); done = np.linalg.norm(hand - goal) < 0.004
        elif name == "DESCEND":
            goal, v = above(s.obj), V_FINE; done = np.linalg.norm(hand - goal) < 0.003
        elif name == "CLOSE":
            done = s.held_obj and self.steps_in_stage >= 1
            if not s.held_obj and self.steps_in_stage > 10:          # closed on nothing
                self._go(retry, t); return self.act(hand, R_hand, grip_obs, s, t)
        elif name == "LIFT":
            goal, v = np.r_[hand[:2], carry], V_FINE; done = hand[2] >= carry - 0.004
        elif name == "TRANSFER":
            xy = hand[:2] + (s.target[:2] - s.obj[:2])                # put the OBJECT over the target
            goal = np.r_[xy, carry]; done = np.linalg.norm(s.target[:2] - s.obj[:2]) < 0.003
        elif name == "LOWER":
            rest = {2: 0.012 + 0.004, 4: 0.005, 5: 0.025 + 0.01}[self.task]   # object centre above the target surface
            xy = hand[:2] + (s.target[:2] - s.obj[:2])
            goal, v = np.r_[xy, hand[2] + (s.target[2] + rest - s.obj[2])], V_FINE
            done = abs(s.target[2] + rest - s.obj[2]) < 0.003 and np.linalg.norm(s.target[:2] - s.obj[:2]) < 0.006
        elif name in ("RELEASE", "LETGO", "REOPEN"):
            done = grip_obs > 0.9 and not s.held_obj and not s.held_handle and self.steps_in_stage >= 1
        elif name == "RISE":
            goal = np.r_[hand[:2], carry]; done = hand[2] >= carry - 0.005 or self.steps_in_stage > 40
        elif name == "H_REACH":
            goal = above(handle, 0.05); done = np.linalg.norm(hand - goal) < 0.004
        elif name == "H_DESCEND":
            goal, v = above(handle), V_FINE; done = np.linalg.norm(hand - goal) < 0.003
        elif name == "GRIP":
            done = s.held_handle and self.steps_in_stage >= 1
            if not s.held_handle and self.steps_in_stage > 10:
                self._go(retry, t); return self.act(hand, R_hand, grip_obs, s, t)
        elif name == "PULL":
            goal, v = hand + [0.0, DRAWER_MAX_OPEN - s.drawer, 0.0], V_HANDLE; done = s.drawer >= DRAWER_MAX_OPEN - 0.004
        elif name == "PUSH":
            goal, v = hand + [0.0, -s.drawer - 0.003, 0.0], V_HANDLE; done = s.drawer <= 0.0005
        elif name == "TURN":
            self.R_goal = A.euler_xyz_to_rot([0.0, 0.0, np.pi / 2]) @ self.R_nom
            done = abs(s.valve) >= 89.5
        elif name == "SWING":
            goal, v = self._lid_goal(hand, s), 0.006; done = s.lid <= 1.0
        elif name == "DONE":
            return np.r_[np.zeros(6), 1.0], "DONE"

        if done:
            if name == "REOPEN":
                self._go(self.reopen_to, t); self.reopen_to = None
            else:
                self._go(self.i + 1, t)
            return self.act(hand, R_hand, grip_obs, s, t)

        move = np.zeros(3) if goal is None else move_toward(hand, goal, v)
        limit = TURN_RATE if name == "TURN" else 0.01
        rot = np.clip(A.rot_to_euler_xyz(self.R_goal @ np.asarray(R_hand, float).T), -limit, limit)
        grip = 1.0 if name in OPEN_STAGES else 0.0
        self.steps_in_stage += 1
        return np.concatenate([move, rot, [grip]]), name
