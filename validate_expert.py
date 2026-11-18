"""
validate_expert.py -- standalone test of the scripted expert (simulator only, no GPU, no model).

    .\\.venv\\Scripts\\python.exe validate_expert.py                       # held-out 12 + 30 random, clean and noisy
    .\\.venv\\Scripts\\python.exe validate_expert.py --random 60 --sigma 0.002

For every scenario the expert runs twice: clean, and with DART-style noise (correlated random drift added to the
executed movement; labels stay clean). Success = expert finished, cube within 15 mm of the marker, resting on the
table, and lifted > 5 cm. Results are printed and saved to closed_loop_results/expert_validation/summary.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import closed_loop_panda as C
import expert_policy as E

ROOT = Path(__file__).resolve().parent
HELD_OUT = list(range(380, 392))
START_X, START_Y = (0.40, 0.49), (-0.10, 0.10)       # same ranges as the demo recorders
MARK_X, MARK_Y = (0.57, 0.66), (-0.20, 0.20)


def random_scene(rng):
    s = (float(rng.uniform(*START_X)), float(rng.uniform(*START_Y)))
    while True:
        t = (float(rng.uniform(*MARK_X)), float(rng.uniform(*MARK_Y)))
        if np.hypot(s[0] - t[0], s[1] - t[1]) >= 0.13:
            return s, t


def summarize(name, rows):
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    err = np.array([r["final_err_m"] for r in rows]) * 1000
    steps = np.array([r["steps"] for r in rows])
    rec = sum(r["recoveries"] for r in rows)
    with_rec = sum(r["recoveries"] > 0 for r in rows)
    print(f"{name:34s} success {ok:3d}/{n:<3d} | final error mm: median {np.median(err):5.1f} max {err.max():6.1f} | "
          f"steps: mean {steps.mean():5.1f} max {steps.max():3d} | recoveries {rec} (in {with_rec} episodes)")
    return dict(name=name, n=n, success=int(ok), median_err_mm=float(np.median(err)), max_err_mm=float(err.max()),
                mean_steps=float(steps.mean()), recoveries=int(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=30, help="number of extra random scenarios")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sigma", type=float, default=0.0013, help="noise innovation std (m); stationary std ~ sigma/0.44")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--skip_clean", action="store_true", help="only run the noisy mode")
    ap.add_argument("--scene", choices=["marker", "bas"], default="marker", help="bas = microgravity lab (bas_lab/bas_lab.xml)")
    ap.add_argument("--out_name", default="summary.json", help="file name inside closed_loop_results/expert_validation/")
    args = ap.parse_args()

    env = C.PandaEnv(C.BAS_SCENE if args.scene == "bas" else C.MARKER_SCENE)
    rng = np.random.default_rng(args.seed)
    scenes = []
    for ep in HELD_OUT:
        _, s, t = C.load_scenario(ep)
        scenes.append((f"held-out {ep}", tuple(s), tuple(t)))
    for i in range(args.random):
        s, t = random_scene(rng)
        scenes.append((f"random {i}", s, t))

    results = {"clean": [], "noisy": []}
    failures = []
    t0 = time.time()
    for mode, sigma in (("clean", 0.0), ("noisy", args.sigma)):
        if mode == "clean" and args.skip_clean:
            continue
        for k, (name, s, t) in enumerate(scenes):
            r = E.run_episode(env, s, t, noise_sigma=sigma, rng=np.random.default_rng(1000 + k), max_steps=args.max_steps)
            drift = np.linalg.norm(np.cumsum(np.array(r["executed"])[:, :3] - np.array(r["labels"])[:, :3], axis=0), axis=1).max() * 100 \
                if r["steps"] else 0.0
            r["noise_cum_cm"] = float(drift)
            results[mode].append(r)
            if not r["success"]:
                last = r["phases"][-1] if r["phases"] else None
                failures.append((mode, name, r["status"], r["steps"], round(r["final_err_m"] * 100, 1), r["recoveries"], last))
        print(f"[{mode}] finished {len(scenes)} scenarios ({time.time() - t0:.0f} s)", flush=True)

    print("\n=== expert validation ===")
    out = []
    held = slice(0, len(HELD_OUT))
    if results["clean"]:
        out.append(summarize("clean, 12 held-out scenes", results["clean"][held]))
        out.append(summarize("clean, random scenes", results["clean"][len(HELD_OUT):]))
    out.append(summarize(f"noisy (sigma {args.sigma * 1000:.1f} mm), held-out", results["noisy"][held]))
    out.append(summarize(f"noisy (sigma {args.sigma * 1000:.1f} mm), random", results["noisy"][len(HELD_OUT):]))
    drift = np.array([r["noise_cum_cm"] for r in results["noisy"]])
    print(f"noise pushed the arm off its intended path by up to (cumulative upper bound): median {np.median(drift):.1f} cm, max {drift.max():.1f} cm")
    lab = np.concatenate([np.array(r["labels"])[:, :3] for r in results["noisy"] if r["steps"]])
    print(f"labels: mean |dpos| {np.linalg.norm(lab, axis=1).mean() * 1000:.2f} mm, max {np.linalg.norm(lab, axis=1).max() * 1000:.2f} mm "
          f"(demos: mean 4.6, max 20.7)")
    if failures:
        print(f"\nfailures ({len(failures)}): mode, scene, status, steps, final error cm, recoveries, last phase change")
        for f in failures[:15]:
            print("  ", f)
    d = ROOT / "closed_loop_results" / "expert_validation"
    d.mkdir(parents=True, exist_ok=True)
    (d / args.out_name).write_text(json.dumps({"summary": out, "failures": failures, "sigma": args.sigma}, indent=2, default=str))
    print(f"\nsaved {d / args.out_name}")
    env.close()


if __name__ == "__main__":
    main()
