"""
Training export from Unreal-recorded expert episodes (demo_recorder.py).

Same output format and labels as build_lab_cam_export.py: soft-relabelled clean expert actions, every
--stride-th frame, normalised with existing statistics (default: the side_v12 model's) so fine-tuning can
continue from that adapter. Images are already 224x224 from Unreal's policy camera.

    python build_unreal_export.py --demo_dir demonstrations_unreal --out panda_openvla_data_unreal_t1
"""
import argparse
import json
from pathlib import Path

import numpy as np

import expert_policy as E
import panda_openvla_dataset as P

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", type=Path, default=ROOT / "demonstrations_unreal")
    ap.add_argument("--stats_from", type=Path, default=ROOT / "model_stats" / "side_v12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    stats_doc = json.loads((args.stats_from / "dataset_statistics.json").read_text())
    stats = stats_doc[P.DATASET_NAME]["action"]
    paths = sorted(args.demo_dir.glob("expert_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no expert_*.npz in {args.demo_dir}")
    imgs, a7, ep_idx, frm_idx, instr, info = [], [], [], [], [], []
    for k, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        lab = E.relabel_soft(d["hand"], d["cube"], d["phase"], d["labels7"])
        idx = np.arange(0, len(lab), args.stride)
        imgs.append(d["images"][idx]); a7.append(lab[idx])
        ep_idx += [k] * len(idx); frm_idx += idx.tolist(); instr += [str(d["instruction"])] * len(idx)
        info.append({"episode": k, "file": p.name, "frames": int(len(idx)), "instruction": str(d["instruction"]),
                     "split": "train", "source": "unreal_expert", "sigma_m": float(d["sigma"])})
    imgs = np.concatenate(imgs); a7 = np.concatenate(a7).astype(np.float32)
    ep_all = np.asarray(ep_idx, dtype=np.int32); frm_all = np.asarray(frm_idx, dtype=np.int32)

    np.save(args.out / "images_new.npy", imgs)
    prev = np.concatenate([[a7[0, 6]], a7[:-1, 6]])
    new_ep = np.concatenate([[True], ep_all[1:] != ep_all[:-1]])
    gchg = (~new_ep) & (np.abs(a7[:, 6] - prev) > 1e-6)
    idle = (np.linalg.norm(a7[:, :3], axis=1) < P.IDLE_POS_M) & (np.abs(a7[:, 3:6]).max(1) < P.IDLE_ROT) & ~gchg
    np.savez_compressed(args.out / "data.npz", actions7=a7, actions_norm=P.normalize_actions(a7, stats),
                        episode_index=ep_all, frame_index=frm_all, idle=idle)
    (args.out / "images_parts.json").write_text(json.dumps([{"path": "images_new.npy", "n": int(len(a7))}]))
    (args.out / "instructions.json").write_text(json.dumps(instr))
    (args.out / "split.json").write_text(json.dumps(
        {"train": [e["episode"] for e in info], "val": [], "episodes": info,
         "build": {"source": "unreal", "camera": "BASPolicyCameraActor 224x224", "stride": args.stride,
                   "soft_relabel": True, "stats_from": str(args.stats_from)}}, indent=1))
    (args.out / "dataset_statistics.json").write_text(json.dumps(stats_doc, indent=2))
    clipped = float(np.mean(np.abs(P.normalize_actions(a7, stats)[:, :6]) >= 1.0))
    print(f"wrote {args.out}: {len(a7)} frames from {len(info)} episodes, images {imgs.shape} "
          f"= {imgs.nbytes / 2**30:.2f} GiB; {100 * clipped:.1f}% of action values at the stats bounds")


if __name__ == "__main__":
    main()
