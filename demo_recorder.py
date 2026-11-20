"""DART-style expert demonstrations recorded from Unreal, through policy_ik_gateway.py --record.

Same recipe and file format as record_expert_demos.py (MuJoCo): each episode draws a noise strength; the
executed hand motion is the expert's plus correlated drift, while the stored label is the CLEAN expert action
for the state actually reached. Images are Unreal's 224x224 policy-camera frames.

Kept only if the expert finished (DONE), the cube ends within 4 cm of the marker, it was lifted > 5 cm, and the
episode has > 50 steps. Output: <dir>/expert_NNNN.npz + manifest_seed<seed>.json. Never overwrites.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from expert_policy import CUBE_REST_Z, NOISE_SCALE

NOISE_RHO = 0.9


class DemoRecorder:
    def __init__(self, out_dir, seed, sigma_min=0.0003, sigma_max=0.0009, clean_frac=0.10):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out / f"manifest_seed{seed}.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"{self.manifest_path} exists; use another --seed")
        self.rng = np.random.default_rng(seed)
        self.sigma_range, self.clean_frac = (sigma_min, sigma_max), clean_frac
        self.manifest = {"created": datetime.now().isoformat(timespec="seconds"), "seed": seed,
                         "source": "unreal", "sigma_range_m": [sigma_min, sigma_max], "clean_frac": clean_frac,
                         "episodes": [], "rejected": []}
        self.ep = None

    def begin(self, episode, instruction, cube, marker_xy):
        if self.ep is not None and not self.ep["finished"]:
            self._reject("not finished (Unreal ended the episode first)")
        sigma = 0.0 if self.rng.random() < self.clean_frac else float(self.rng.uniform(*self.sigma_range))
        self.ep = {"episode": episode, "instruction": instruction, "sigma": sigma, "noise": np.zeros(3),
                   "start_xy": np.asarray(cube[:2], np.float32), "target_xy": np.asarray(marker_xy, np.float32),
                   "finished": False, "t0": time.time(),
                   **{k: [] for k in ("images", "labels", "executed", "phase", "hand", "cube", "grip_obs")}}

    def step(self, rgb, hand, cube, grip, label, phase):
        """Record one step; return the action to EXECUTE (label + drift). After DONE, nothing is recorded."""
        ep = self.ep
        if ep is None or ep["finished"]:
            return label
        if phase == "DONE":
            self._finish(cube)
            return label
        executed = np.array(label, dtype=np.float64)
        if ep["sigma"] > 0 and NOISE_SCALE[phase] > 0:
            ep["noise"] = NOISE_RHO * ep["noise"] + ep["sigma"] * self.rng.normal(size=3)
            executed[:3] += NOISE_SCALE[phase] * ep["noise"]
        else:
            ep["noise"] = NOISE_RHO * ep["noise"]
        ep["images"].append(np.frombuffer(rgb, np.uint8).reshape(224, 224, 3))
        ep["labels"].append(label); ep["executed"].append(executed); ep["phase"].append(phase)
        ep["hand"].append(hand); ep["cube"].append(cube); ep["grip_obs"].append(grip)
        return executed.tolist()

    def _info(self, ep, status, err):
        return {"episode": ep["episode"], "start_xy": ep["start_xy"].tolist(), "target_xy": ep["target_xy"].tolist(),
                "sigma_m": ep["sigma"], "steps": len(ep["labels"]), "final_err_m": err, "status": status}

    def _finish(self, cube):
        ep = self.ep
        ep["finished"] = True
        err = float(np.linalg.norm(np.asarray(cube[:2]) - ep["target_xy"]))
        lifted = max(c[2] for c in ep["cube"]) - CUBE_REST_Z > 0.05 if ep["cube"] else False
        info = self._info(ep, "done", err)
        if not (err < 0.04 and lifted and len(ep["labels"]) > 50):
            return self._reject(f"err {err * 1000:.0f} mm, lifted {lifted}, steps {len(ep['labels'])}", info)
        nums = [int(p.stem.split("_")[-1]) for p in self.out.glob("expert_*.npz")]
        path = self.out / f"expert_{max(nums, default=-1) + 1:04d}.npz"
        np.savez_compressed(
            path, images=np.stack(ep["images"]),
            labels7=np.asarray(ep["labels"], np.float32), executed7=np.asarray(ep["executed"], np.float32),
            phase=np.asarray(ep["phase"]), hand=np.asarray(ep["hand"], np.float32),
            cube=np.asarray(ep["cube"], np.float32), grip_obs=np.asarray(ep["grip_obs"], np.float32),
            start_xy=ep["start_xy"], target_xy=ep["target_xy"],
            instruction=np.asarray(ep["instruction"]), sigma=np.asarray(ep["sigma"], np.float32))
        info["file"] = path.name
        self.manifest["episodes"].append(info)
        self._save_manifest()
        print(f"SAVED {path.name}: {info['steps']} steps, err {err * 1000:.1f} mm, sigma {ep['sigma'] * 1000:.2f} mm, "
              f"{time.time() - ep['t0']:.0f}s | total saved {len(self.manifest['episodes'])}", flush=True)

    def _reject(self, why, info=None):
        ep = self.ep
        ep["finished"] = True
        info = info or self._info(ep, "unfinished", None)
        info["reason"] = why
        self.manifest["rejected"].append(info)
        self._save_manifest()
        print(f"REJECTED episode {ep['episode']}: {why}", flush=True)

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
