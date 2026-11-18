"""
record_expert_demos.py -- record DART-style corrective data with the scripted expert (simulator only).

Each episode: a random scene (same ranges as the demo recorders), the expert runs with correlated random drift
added to the executed movement (strength drawn per episode; some episodes clean), and at EVERY step we save the
camera image of the state actually reached together with the expert's CLEAN action for that state. Training on
this teaches the model what to do when it is slightly off the ideal path.

An episode is kept only if the expert finished and the cube ends within 4 cm of the marker, resting on the table,
having been lifted > 5 cm. Output (never overwrites): demonstrations_expert/expert_NNNN.npz + expert_manifest_seed*.json

    .\\.venv\\Scripts\\python.exe record_expert_demos.py --count 3 --seed 11        # smoke test
    .\\.venv\\Scripts\\python.exe record_expert_demos.py --count 220 --seed 5000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import closed_loop_panda as C
import expert_policy as E
import validate_expert as V

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "demonstrations_expert"
INSTRUCTION = "Pick up the red cube and place it on the yellow marker."


def biased_scene(rng, far_frac, far_min):
    """With probability far_frac the marker is placed at |y| >= far_min (either side, x uniform); otherwise the
    usual uniform scene. The cube start and the 13 cm minimum distance are as in validate_expert.random_scene."""
    if rng.random() >= far_frac:
        return V.random_scene(rng)
    while True:
        s = (float(rng.uniform(*V.START_X)), float(rng.uniform(*V.START_Y)))
        side = -1.0 if rng.random() < 0.5 else 1.0
        t = (float(rng.uniform(*V.MARK_X)), float(side * rng.uniform(far_min, V.MARK_Y[1])))
        if np.hypot(s[0] - t[0], s[1] - t[1]) >= 0.13:
            return s, t


def next_index():
    OUT.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem.split("_")[-1]) for p in OUT.glob("expert_*.npz")]
    return max(nums, default=-1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=220, help="scenarios to attempt")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--sigma_min", type=float, default=0.0003, help="noise innovation std range (m)")
    ap.add_argument("--sigma_max", type=float, default=0.0009)
    ap.add_argument("--clean_frac", type=float, default=0.10, help="fraction of episodes without noise")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--far_y_frac", type=float, default=0.0,
                    help="fraction of scenes whose marker is at |y| >= --far_y_min (targets the model's weak zone)")
    ap.add_argument("--far_y_min", type=float, default=0.12)
    ap.add_argument("--scene", default=None, help="scene xml (default: the marker scene); e.g. bas_lab/bas_lab.xml")
    ap.add_argument("--out_dir", default=None, help="output folder (default demonstrations_expert/)")
    args = ap.parse_args()
    global OUT
    if args.out_dir:
        OUT = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    manifest_path = OUT / f"expert_manifest_seed{args.seed}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path.name} exists; use another --seed (no overwriting).")

    scene = C.MARKER_SCENE if args.scene is None else Path(args.scene)
    scene = scene if scene.is_absolute() else ROOT / scene
    env = C.PandaEnv(scene)
    rng = np.random.default_rng(args.seed)
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
                "count_requested": args.count, "instruction": INSTRUCTION,
                "sigma_range_m": [args.sigma_min, args.sigma_max], "clean_frac": args.clean_frac,
                "far_y_frac": args.far_y_frac, "far_y_min": args.far_y_min, "scene": str(scene.name),
                "episodes": [], "rejected": []}
    t_start, saved = time.time(), 0
    for k in range(args.count):
        start_xy, target_xy = biased_scene(rng, args.far_y_frac, args.far_y_min)
        sigma = 0.0 if rng.random() < args.clean_frac else float(rng.uniform(args.sigma_min, args.sigma_max))
        t0 = time.time()
        r = E.run_episode(env, start_xy, target_xy, noise_sigma=sigma, rng=np.random.default_rng(args.seed * 1000 + k),
                          max_steps=args.max_steps, render=True)
        keep = r["status"] == "done" and r["final_err_m"] < 0.04 and r["lifted"] and r["steps"] > 50
        info = {"scenario_index": k, "start_xy": list(start_xy), "target_xy": list(target_xy), "sigma_m": sigma,
                "steps": r["steps"], "final_err_m": r["final_err_m"], "recoveries": r["recoveries"], "status": r["status"]}
        if keep:
            idx = next_index()
            path = OUT / f"expert_{idx:04d}.npz"
            np.savez_compressed(
                path,
                images=np.asarray(r["images"], dtype=np.uint8),
                labels7=np.asarray(r["labels"], dtype=np.float32),       # clean expert actions (training labels)
                executed7=np.asarray(r["executed"], dtype=np.float32),   # what was actually applied (with noise)
                phase=np.asarray(r["phase"]),
                hand=np.asarray(r["hand"], dtype=np.float32), cube=np.asarray(r["cube"], dtype=np.float32),
                grip_obs=np.asarray(r["grip_obs"], dtype=np.float32),
                start_xy=np.asarray(start_xy, dtype=np.float32), target_xy=np.asarray(target_xy, dtype=np.float32),
                instruction=np.asarray(INSTRUCTION), sigma=np.asarray(sigma, dtype=np.float32))
            info["file"] = path.name
            manifest["episodes"].append(info)
            saved += 1
        else:
            manifest["rejected"].append(info)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[{k + 1}/{args.count}] {'saved ' + info['file'] if keep else 'REJECTED ' + r['status']} | sigma {sigma * 1000:.2f} mm | "
              f"{r['steps']} steps | err {r['final_err_m'] * 1000:.1f} mm | {time.time() - t0:.1f}s | "
              f"elapsed {(time.time() - t_start) / 60:.1f} min | saved {saved}", flush=True)
    env.close()
    print(f"\nDONE: saved {saved}/{args.count}, rejected {len(manifest['rejected'])}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
