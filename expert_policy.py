"""
expert_policy.py -- closed-loop scripted expert for the marker pick-and-place task.

Unlike the open-loop demo recorder (fixed waypoints planned once), this expert looks at the CURRENT state at
every step and decides the next hand movement, so it can finish the task from any situation, including
slightly wrong ones. It reads the true cube / hand / marker positions from the simulator (privileged
information the learned model does not have). It is not an AI and needs no training.

Output per step: the same 7-D action the model predicts,
    [dx, dy, dz, droll, dpitch, dyaw, gripper]    (world-frame hand deltas in metres/radians; gripper 1=open 0=closed)

Phases (a small state machine):
    REACH    move above the cube (rise first if the hand is low and clearly misaligned)
    DESCEND  go down to grasp height, tracking the cube's xy. REACH and DESCEND use ONE smooth law
             (`approach_move`: goal height rises continuously with the xy error); the two names are only labels.
    CLOSE    stand still, command the gripper closed until the fingers have shut on the cube
    LIFT     go straight up to carrying height
    TRANSFER move so that the CUBE (not the hand) is above the marker
    LOWER    go down to placing height, keeping the cube aligned with the marker
    RELEASE  stand still, open the gripper
    RETREAT  go straight up, then DONE
    REOPEN   recovery: the cube was dropped / the grasp missed -> open the gripper and start again at REACH

`run_episode` executes the expert in a PandaEnv (closed_loop_panda), optionally with DART-style noise: the
executed movement is the expert's plus correlated random drift, but the recorded LABEL is always the clean
expert action computed from the state the noise led to.
"""

import numpy as np

import openvla_adapter as A

# heights of the HAND body (metres)
Z_HOVER = 0.62
Z_GRASP = 0.494       # fingers around the cube (demos close at 0.494)
Z_CARRY = 0.60
Z_PLACE = 0.50
Z_RETREAT = 0.64
CUBE_REST_Z = 0.40    # cube centre resting on the table

# speeds, metres per 0.1 s step (demos average 3.5-4.7 mm)
V_REACH, V_DESC, V_LIFT, V_TRANS, V_LOWER, V_RETREAT = 0.005, 0.004, 0.004, 0.005, 0.004, 0.004

# how much injected noise each phase tolerates (0 = none: the arm must stay still or is near contact)
NOISE_SCALE = {"REACH": 1.0, "DESCEND": 0.4, "CLOSE": 0.0, "LIFT": 0.4, "TRANSFER": 1.0,
               "LOWER": 0.4, "RELEASE": 0.0, "RETREAT": 0.0, "REOPEN": 0.0, "DONE": 0.0}
GRIPPER_OPEN_PHASES = {"REACH", "DESCEND", "RELEASE", "RETREAT", "REOPEN", "DONE"}


# soft approach law (D62): the goal height depends continuously on the horizontal error to the cube, so that a
# picture that looks the same (cube ~1 cm off-centre under the gripper) always gets the same kind of label.
# Hard switches at one offset made "go up" and "go down" labels coexist for nearly identical images -> hovering.
RAMP_LO, RAMP_HI = 0.006, 0.030     # horizontal error (m): at/below LO descend fully, at/above HI stay at hover height


def approach_move(hand, cxy):
    """Hand step towards the cube for REACH/DESCEND: xy towards the cube, z towards a goal that rises with the xy error."""
    hand = np.asarray(hand, float)
    exy = float(np.linalg.norm(np.asarray(cxy, float) - hand[:2]))
    ramp = float(np.clip((exy - RAMP_LO) / (RAMP_HI - RAMP_LO), 0.0, 1.0))
    z_goal = Z_GRASP + ramp * (Z_HOVER - Z_GRASP)
    v = V_DESC + (V_REACH - V_DESC) * ramp
    if hand[2] < Z_HOVER - 0.025 and exy > RAMP_HI:   # low and clearly misaligned: rise first, do not drag the cube
        return move_toward(hand, [hand[0], hand[1], Z_HOVER], V_REACH)
    return move_toward(hand, [cxy[0], cxy[1], z_goal], v)


def move_toward(pos, goal, vmax):
    """Step of at most vmax towards goal; lands exactly on the goal when it is closer than vmax."""
    d = np.asarray(goal, dtype=float) - np.asarray(pos, dtype=float)
    n = float(np.linalg.norm(d))
    return d if n <= vmax else d * (vmax / n)


def relabel_soft(hand, cube, phase, labels):
    """Relabel a stored episode with the soft approach law (D62): REACH/DESCEND frames get the xyz label that
    `approach_move` gives for the stored hand/cube positions; every other frame (and rotation/gripper) is kept."""
    out = np.array(labels, dtype=np.float32, copy=True)
    for t, ph in enumerate(phase):
        if ph in ("REACH", "DESCEND"):
            out[t, :3] = approach_move(hand[t], cube[t][:2])
    return out


class ExpertPolicy:
    def __init__(self, target_xy, nominal_rot):
        self.target = np.asarray(target_xy, dtype=float)
        self.R_nom = np.asarray(nominal_rot, dtype=float)
        self.phase = "REACH"
        self.steps_in_phase = 0
        self.recoveries = 0
        self.history = []   # (step index, phase) at every phase change
        self.events = []    # why a recovery started: (step, reason, details)

    def _go(self, phase, t):
        self.phase, self.steps_in_phase = phase, 0
        self.history.append((t, phase))

    def act(self, hand, cube, grip_obs, R_hand, t=0):
        """hand/cube: xyz positions; grip_obs: measured opening 0..1; R_hand: 3x3 hand rotation. Returns (a7, phase)."""
        hand, cube = np.asarray(hand, float), np.asarray(cube, float)
        hxy, cxy = hand[:2], cube[:2]
        exy = float(np.linalg.norm(cxy - hxy))
        ph = self.phase

        # ---- phase transitions (evaluated on the current observation) ----
        if ph in ("LIFT", "TRANSFER", "LOWER"):
            # Cube lost = it left the fingers. Physical sensor: a gripper holding the 5 cm cube reads ~0.62 open;
            # if the cube slips out the fingers close to ~0. (Geometric tests on the cube height are unreliable:
            # the cube creeps down in the fingers by up to ~4 cm while carried, and it legitimately descends
            # with the hand while lowering.) Also: cube far from the hand in xy, or hanging implausibly low.
            hanging = hand[2] - cube[2]
            if grip_obs < 0.45 or exy > 0.05 or (hanging > 0.2 and cube[2] > CUBE_REST_Z + 0.02):
                self.events.append((t, ph, "cube lost", round(float(grip_obs), 2), round(exy, 3)))
                self._go("REOPEN", t); ph = "REOPEN"; self.recoveries += 1
        if ph == "REACH":
            if exy < 0.012:                           # same movement law as DESCEND: the name is only a label
                self._go("DESCEND", t); ph = "DESCEND"
        elif ph == "DESCEND":
            if exy > RAMP_HI and hand[2] > Z_GRASP + 0.03:
                self._go("REACH", t); ph = "REACH"
            elif hand[2] <= Z_GRASP + 0.003 and exy < 0.008:
                self._go("CLOSE", t); ph = "CLOSE"
        elif ph == "CLOSE":
            if grip_obs < 0.30:                       # closed on nothing
                self.events.append((t, ph, "closed on nothing", round(float(grip_obs), 2), round(exy, 3)))
                self._go("REOPEN", t); ph = "REOPEN"; self.recoveries += 1
            elif grip_obs < 0.75 and self.steps_in_phase >= 1:   # fingers are on the cube
                self._go("LIFT", t); ph = "LIFT"
        elif ph == "LIFT":
            if hand[2] >= Z_CARRY - 0.004:
                self._go("TRANSFER", t); ph = "TRANSFER"
        elif ph == "TRANSFER":
            if np.linalg.norm(self.target - cxy) < 0.003:
                self._go("LOWER", t); ph = "LOWER"
        elif ph == "LOWER":
            if hand[2] <= Z_PLACE + 0.003 and np.linalg.norm(self.target - cxy) < 0.006:
                self._go("RELEASE", t); ph = "RELEASE"
        elif ph == "RELEASE":
            if grip_obs > 0.9 and self.steps_in_phase >= 1:
                self._go("RETREAT", t); ph = "RETREAT"
        elif ph == "RETREAT":
            if hand[2] >= Z_RETREAT - 0.005 or self.steps_in_phase > 40:
                self._go("DONE", t); ph = "DONE"
        elif ph == "REOPEN":
            if grip_obs > 0.9 and self.steps_in_phase >= 1:
                self._go("REACH", t); ph = "REACH"

        # ---- movement for the (possibly new) phase ----
        move = np.zeros(3)
        if ph in ("REACH", "DESCEND"):
            move = approach_move(hand, cxy)
        elif ph == "LIFT":
            move = move_toward(hand, [hxy[0], hxy[1], Z_CARRY], V_LIFT)
        elif ph in ("TRANSFER", "LOWER"):
            goal_xy = hxy + (self.target - cxy)          # put the CUBE, not the hand, on the marker
            if ph == "TRANSFER":
                move = move_toward(hand, [goal_xy[0], goal_xy[1], Z_CARRY], V_TRANS)
            else:
                move = move_toward(hand, [goal_xy[0], goal_xy[1], Z_PLACE], V_LOWER)
        elif ph == "RETREAT":
            move = move_toward(hand, [hxy[0], hxy[1], Z_RETREAT], V_RETREAT)
        # CLOSE / RELEASE / REOPEN / DONE: stand still

        rot = np.clip(A.rot_to_euler_xyz(self.R_nom @ np.asarray(R_hand, float).T), -0.01, 0.01)
        a7 = np.concatenate([move, rot, [1.0 if ph in GRIPPER_OPEN_PHASES else 0.0]])
        self.steps_in_phase += 1
        return a7, ph


def run_episode(env, start_xy, target_xy, noise_sigma=0.0, noise_rho=0.9, rng=None, max_steps=400,
                render=False, log_every=0):
    """
    Run the expert in `env` (closed_loop_panda.PandaEnv on the marker scene).
    noise_sigma: std (metres) of the innovation of an Ornstein-Uhlenbeck drift added to the executed hand
                 movement (stationary std = sigma / sqrt(1 - rho^2)); 0 = clean expert.
    render: also return the 320x240 camera image of every step (for recording).
    Returns a dict: success, status, steps, final_err_m, recoveries, phases, and per-step arrays
    (hand, cube, grip_obs, labels = clean expert actions, executed = what was applied, phase names).
    """
    rng = rng or np.random.default_rng(0)
    env.reset(start_xy, target_xy)
    R_nom = env.data.xmat[env.hand_id].reshape(3, 3).copy()
    pol = ExpertPolicy(target_xy, R_nom)
    n = np.zeros(3)
    out = {k: [] for k in ("hand", "cube", "grip_obs", "labels", "executed", "phase")}
    if render:
        out["images"] = []
    status = "max_steps"
    for t in range(max_steps):
        hand, cube = env.hand(), env.cube()
        state = env.state()
        R_hand = env.data.xmat[env.hand_id].reshape(3, 3)
        a_exp, ph = pol.act(hand, cube, state[7], R_hand, t)
        if ph == "DONE":
            status = "done"
            break
        if render:
            out["images"].append(env.image())
        a_exec = a_exp.copy()
        if noise_sigma > 0 and NOISE_SCALE[ph] > 0:
            n = noise_rho * n + noise_sigma * rng.normal(size=3)
            a_exec[:3] += NOISE_SCALE[ph] * n
        else:
            n = noise_rho * n   # let the drift decay while the arm must stay clean
        env.execute_delta(a_exec)
        out["hand"].append(hand); out["cube"].append(cube); out["grip_obs"].append(state[7])
        out["labels"].append(a_exp); out["executed"].append(a_exec); out["phase"].append(ph)
        if log_every and t % log_every == 0:
            print(f"   t={t:3d} {ph:8s} hand={np.round(hand, 3)} cube={np.round(cube, 3)} grip={state[7]:.2f}")
    cube = env.cube()
    err = float(np.linalg.norm(cube[:2] - np.asarray(target_xy)))
    lifted = (np.max([c[2] for c in out["cube"]] + [cube[2]]) - CUBE_REST_Z) > 0.05 if out["cube"] else False
    ok = bool(status == "done" and err < 0.015 and abs(cube[2] - CUBE_REST_Z) < 0.02 and lifted)
    out.update(success=ok, status=status, steps=len(out["labels"]), final_err_m=err,
               recoveries=pol.recoveries, phases=pol.history, events=pol.events, lifted=bool(lifted))
    return out
