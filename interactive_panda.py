"""
interactive_panda.py -- type a command, watch the Panda do it (live MuJoCo window).

Prerequisites: policy_server.py must be running and reachable at http://127.0.0.1:8000
(same machine, or a port-forward from the GPU machine).

    .\\.venv\\Scripts\\python.exe interactive_panda.py

What the model understands
  The model was trained on ONE instruction:
      "Pick up the red cube and place it on the yellow marker."
  It has never seen other wording, so a different sentence is passed to it but will most likely be ignored.
  What it does react to is the scene: where the red cube and the yellow marker are.

Commands (type at the 'command>' prompt)
  <the instruction sentence>   run the model on the current scene   (Enter alone also runs it)
  random                        new random cube / marker positions
  cube X Y                      put the cube at (X, Y)      training range: X 0.40..0.49, Y -0.10..0.10
  marker X Y                    put the marker at (X, Y)    training range: X 0.57..0.66, Y -0.20..0.20
  show                          print the current scene
  help | quit

Notes: each policy step takes about 1 s (one 7-billion-parameter model call per step), so the arm
moves in short bursts; a full pick-and-place takes about 3-4 minutes. Ctrl+C stops the current run.
"""

import argparse
import sys
import time

import numpy as np

import closed_loop_panda as C
import panda_openvla_dataset as P

TRAINED_INSTRUCTION = "Pick up the red cube and place it on the yellow marker."
CUBE_X, CUBE_Y = (0.40, 0.49), (-0.10, 0.10)
MARK_X, MARK_Y = (0.57, 0.66), (-0.20, 0.20)


def random_scene(rng):
    start = (float(rng.uniform(*CUBE_X)), float(rng.uniform(*CUBE_Y)))
    for _ in range(200):
        target = (float(rng.uniform(*MARK_X)), float(rng.uniform(*MARK_Y)))
        if np.hypot(start[0] - target[0], start[1] - target[1]) >= 0.13:
            return start, target
    return start, target


def in_range(v, lo_hi):
    return lo_hi[0] <= v <= lo_hi[1]


def run_episode(env, policy, instruction, start_xy, target_xy, viewer, max_steps):
    env.reset(start_xy, target_xy)
    viewer.sync()
    instr = P.format_instruction(instruction)
    held, status = 0, "max_steps"
    t0 = time.time()
    for t in range(max_steps):
        if not viewer.is_running():
            return "viewer_closed", None
        img, state = env.image(), env.state()
        a7, lat = policy(P.resize_image(img), instr)
        n = np.linalg.norm(a7[:3])
        if n > C.MAX_HAND_STEP:  # same safety bound as in the evaluations
            a7 = a7.copy()
            a7[:3] *= C.MAX_HAND_STEP / n
        if not np.all(np.isfinite(a7)):
            return "nonfinite_action", None
        env.execute_delta(a7)
        cube = env.cube()
        if not np.all(np.isfinite(cube)) or cube[2] < C.TABLE_TOP_Z - 0.10:
            return "cube_fell", float(np.linalg.norm(cube[:2] - np.asarray(target_xy)))
        err = float(np.linalg.norm(cube[:2] - np.asarray(target_xy)))
        # released = fingers open wider than the cube (a held 5 cm cube reads ~0.62-0.69, so > 0.5 is not enough)
        placed = (err < C.SUCCESS_XY_M and abs(cube[2] - C.CUBE_REST_Z) < 0.02
                  and env.state()[7] > C.RELEASE_READING and not env.cube_touches_gripper())
        held = held + 1 if placed else 0
        if t % 10 == 0:
            phase = "gripper closed" if a7[6] < 0.5 else "gripper open"
            print(f"   step {t:3d}  hand z={env.hand()[2]:.3f}  cube->marker {err * 100:5.1f} cm  ({phase})", flush=True)
        if held >= C.SUCCESS_HOLD_STEPS:
            status = "success"
            break
    cube = env.cube()
    return status, float(np.linalg.norm(cube[:2] - np.asarray(target_xy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_steps", type=int, default=300)   # success now needs the release, so allow extra steps
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", type=int, default=0,
                    help="open the viewer and move with zero actions for N steps, without the model (self-test)")
    ap.add_argument("--scene", choices=["marker", "bas"], default="marker",
                    help="marker = table scene (default); bas = microgravity space-station lab")
    ap.add_argument("--episode", type=int, default=None,
                    help="start with the cube/marker layout of this held-out episode (e.g. 384) instead of a random one")
    args = ap.parse_args()

    import mujoco.viewer

    env = C.PandaEnv({"marker": C.MARKER_SCENE, "bas": C.BAS_SCENE}[args.scene])
    rng = np.random.default_rng(args.seed)
    if args.episode is not None:
        _, start_xy, target_xy = C.load_scenario(args.episode)
        start_xy, target_xy = np.asarray(start_xy, float), np.asarray(target_xy, float)
    else:
        start_xy, target_xy = random_scene(rng)

    if args.dry_run:
        policy = lambda img, ins: (np.array([0, 0, 0, 0, 0, 0, 1.0]), 0.0)  # noqa: E731
    else:
        try:
            policy = C.ServerPolicy(args.port)
        except Exception as e:
            print(f"Cannot reach the policy server on port {args.port}: {e}\n"
                  "Start policy_server.py (and the port-forward, if any) first.")
            sys.exit(1)
        print("connected to policy server:", policy.info.get("adapter", "?"))

    env.reset(start_xy, target_xy)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.55, 0.0, 0.43]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.65, 135, -25
        counter = {"n": 0}

        def hook():
            counter["n"] += 1
            if counter["n"] % 2 == 0:
                viewer.sync()
                time.sleep(0.004)  # roughly real-time motion (2 ms physics step)

        env.step_hook = hook
        viewer.sync()

        if args.dry_run:
            print(f"dry run: {args.dry_run} policy steps with zero actions ...")
            status, err = run_episode(env, policy, TRAINED_INSTRUCTION, start_xy, target_xy, viewer, args.dry_run)
            print(f"dry run finished ({status}); the viewer and the step hook work.")
            time.sleep(1.0)
            return

        print(__doc__.split("Commands")[0].strip().split("What the model understands")[1].strip().replace("\n  ", "\n"))
        print("\nType 'help' for the commands. Close the window or type 'quit' to exit.\n")
        while viewer.is_running():
            print(f"scene: cube at ({start_xy[0]:.2f}, {start_xy[1]:+.2f}), marker at ({target_xy[0]:.2f}, {target_xy[1]:+.2f})")
            try:
                line = input("command> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            low = line.lower()
            if low in ("quit", "exit", "q"):
                break
            if low in ("help", "?"):
                print(__doc__.split("Commands (type at the 'command>' prompt)")[1].strip())
                continue
            if low == "show":
                continue
            if low == "random":
                start_xy, target_xy = random_scene(rng)
                env.reset(start_xy, target_xy)
                viewer.sync()
                continue
            parts = low.split()
            if parts and parts[0] in ("cube", "marker"):
                try:
                    x, y = float(parts[1]), float(parts[2])
                except (IndexError, ValueError):
                    print("usage: cube X Y   or   marker X Y   (metres)")
                    continue
                rx, ry = (CUBE_X, CUBE_Y) if parts[0] == "cube" else (MARK_X, MARK_Y)
                if not (in_range(x, rx) and in_range(y, ry)):
                    print(f"  warning: outside the training range X {rx}, Y {ry}; the model has never seen this and may fail")
                if parts[0] == "cube":
                    start_xy = (x, y)
                else:
                    target_xy = (x, y)
                env.reset(start_xy, target_xy)
                viewer.sync()
                continue

            instruction = line if line else TRAINED_INSTRUCTION
            if P.format_instruction(instruction) != P.format_instruction(TRAINED_INSTRUCTION):
                print("  note: the model was trained only on 'Pick up the red cube and place it on the yellow marker.'\n"
                      "        Other wording is passed through but will most likely have no effect.")
            print(f"running: \"{instruction}\"  (about 3-4 minutes; Ctrl+C stops)")
            t0 = time.time()
            try:
                status, err = run_episode(env, policy, instruction, start_xy, target_xy, viewer, args.max_steps)
            except KeyboardInterrupt:
                status, err = "stopped by you", None
            except Exception as e:  # e.g. server unreachable
                status, err = f"error: {e}", None
            msg = f"result: {status}"
            if err is not None:
                msg += f"  | final distance cube->marker: {err * 100:.1f} cm"
                msg += "  -> SUCCESS (within 4 cm)" if err < C.SUCCESS_XY_M else "  -> not within 4 cm"
            print(f"{msg}  ({time.time() - t0:.0f} s)\n")
            # leave the final pose on screen; the next run / 'random' resets the scene

    env.close()


if __name__ == "__main__":
    main()
