"""
Training export from Unreal-recorded expert episodes (demo_recorder.py).

Same output format and labels as build_lab_cam_export.py: soft-relabelled clean expert actions, every
--stride-th frame, normalised with existing statistics (default: the side_v12 model's) so fine-tuning can
continue from that adapter. Images are already 224x224 from Unreal's policy camera.
Multitask: task 1 frames are soft-relabelled, tasks 2-5 keep the expert labels; --new_stats computes bounds
from this data; --val_per_task holds out each task's last N episodes as val_frames.npz.

    python build_unreal_export.py --demo_dir demonstrations_unreal --out panda_openvla_data_unreal_t1
    python build_unreal_export.py --demo_dir demonstrations_bench --out panda_openvla_data_bench \
        --stride 3 --new_stats --val_per_task 10
    # same frames, sub-step sentences for tasks 2 and 5, images shared with the export above
    python build_unreal_export.py --demo_dir demonstrations_bench --out panda_openvla_data_bench_sub \
        --stride 3 --new_stats --val_per_task 10 --substeps --images_from panda_openvla_data_bench
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import expert_policy as E
import multitask_expert as M
import openvla_adapter as A
import panda_openvla_dataset as P

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", type=Path, default=ROOT / "demonstrations_unreal")
    ap.add_argument("--stats_from", type=Path, default=ROOT / "model_stats" / "side_v12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--new_stats", action="store_true", help="action bounds from this data, not --stats_from")
    ap.add_argument("--val_per_task", type=int, default=0, help="hold out each task's last N episodes")
    ap.add_argument("--substeps", action="store_true", help="tasks 2 and 5: per-frame sub-step sentences")
    ap.add_argument("--images_from", type=Path, default=None, help="reuse this export's images_new.npy (same frames)")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    stats_doc = json.loads((args.stats_from / "dataset_statistics.json").read_text())
    stats = stats_doc[P.DATASET_NAME]["action"]
    paths = sorted(args.demo_dir.glob("expert_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no expert_*.npz in {args.demo_dir}")
    task_of = {p: int(np.load(p)["task"]) if "task" in np.load(p).files else 1 for p in paths}
    val = set()
    if args.val_per_task:
        for t in set(task_of.values()):
            val |= set([p for p in paths if task_of[p] == t][-args.val_per_task:])
        write_val(sorted(val), task_of, args.out / "val_frames.npz", substeps=args.substeps)
    paths = [p for p in paths if p not in val]
    labels = {p: label(np.load(p, allow_pickle=True), task_of[p]) for p in paths}
    if args.new_stats:
        stats = new_stats(np.concatenate([l[::args.stride] for l in labels.values()]))
        stats_doc = {P.DATASET_NAME: {**stats_doc[P.DATASET_NAME], "action": stats}}
    n_total = sum(len(range(0, len(l), args.stride)) for l in labels.values())
    if args.images_from:
        imgs_path = args.images_from / "images_new.npy"
        imgs = np.load(imgs_path, mmap_mode="r")
        assert imgs.shape[0] == n_total, f"{imgs_path}: {imgs.shape[0]} frames, this export has {n_total}"
    else:
        imgs_path = args.out / "images_new.npy"
        imgs = np.lib.format.open_memmap(imgs_path, "w+", np.uint8, (n_total, 224, 224, 3))
    a7, ep_idx, frm_idx, instr, info = [], [], [], [], []
    for k, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        lab = labels[p]
        idx = np.arange(0, len(lab), args.stride)
        if not args.images_from:
            imgs[len(frm_idx):len(frm_idx) + len(idx)] = d["images"][idx]
        a7.append(lab[idx])
        ep_idx += [k] * len(idx); frm_idx += idx.tolist()
        instr += [sentences(d, task_of[p], args.substeps)[i] for i in idx]
        info.append({"episode": k, "file": p.name, "frames": int(len(idx)), "instruction": str(d["instruction"]),
                     "split": "train", "source": "unreal_expert", "sigma_m": float(d["sigma"]), "task": task_of[p]})
    if not args.images_from:
        imgs.flush()
    a7 = np.concatenate(a7).astype(np.float32)
    ep_all = np.asarray(ep_idx, dtype=np.int32); frm_all = np.asarray(frm_idx, dtype=np.int32)

    prev = np.concatenate([[a7[0, 6]], a7[:-1, 6]])
    new_ep = np.concatenate([[True], ep_all[1:] != ep_all[:-1]])
    gchg = (~new_ep) & (np.abs(a7[:, 6] - prev) > 1e-6)
    idle = (np.linalg.norm(a7[:, :3], axis=1) < P.IDLE_POS_M) & (np.abs(a7[:, 3:6]).max(1) < P.IDLE_ROT) & ~gchg
    np.savez_compressed(args.out / "data.npz", actions7=a7, actions_norm=P.normalize_actions(a7, stats),
                        episode_index=ep_all, frame_index=frm_all, idle=idle)
    rel = os.path.relpath(imgs_path, args.out)
    (args.out / "images_parts.json").write_text(json.dumps([{"path": rel, "n": int(len(a7))}]))
    (args.out / "instructions.json").write_text(json.dumps(instr))
    (args.out / "split.json").write_text(json.dumps(
        {"train": [e["episode"] for e in info], "val": [], "episodes": info,
         "build": {"source": "unreal", "camera": "BASPolicyCameraActor 224x224", "stride": args.stride,
                   "soft_relabel": "task 1", "stats_from": "this data" if args.new_stats else str(args.stats_from),
                   "val_files": [p.name for p in sorted(val)]}}, indent=1))
    (args.out / "dataset_statistics.json").write_text(json.dumps(stats_doc, indent=2))
    clipped = float(np.mean(np.abs(P.normalize_actions(a7, stats)[:, :6]) >= 1.0))
    print(f"wrote {args.out}: {len(a7)} frames from {len(info)} episodes, images {imgs.shape} "
          f"= {imgs.nbytes / 2**30:.2f} GiB; {100 * clipped:.1f}% of action values at the stats bounds")


def label(d, task):
    if task == 1:
        return E.relabel_soft(d["hand"], d["cube"], d["phase"], d["labels7"])
    return np.asarray(d["labels7"], dtype=np.float32)


def new_stats(a7):
    """q01/q99 bounds from the data; rotation bounds fixed (expert hold corrections and the valve turn are rare)."""
    s = A.compute_stats(a7)
    s["q01"][3:6], s["q99"][3:6] = [-0.01, -0.01, -0.04], [0.01, 0.01, 0.04]
    return s


def sentences(d, task, substeps):
    """Instruction per recorded frame: the task sentence, or sub-step sentences for tasks 2 and 5."""
    if substeps and task in M.SUBSTEPS:
        return M.substep_labels(task, d["phase"])
    return [str(d["instruction"])] * len(d["labels7"])


def write_val(paths, task_of, out, stride=2, substeps=False):
    """Held-out frames for eval_checkpoints.py: image, label, phase and instruction per frame."""
    imgs, labs, phase, instr = [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        idx = np.arange(0, len(d["labels7"]), stride)
        imgs.append(d["images"][idx]); labs.append(label(d, task_of[p])[idx]); phase.append(d["phase"][idx])
        instr += [sentences(d, task_of[p], substeps)[i] for i in idx]
    np.savez(out, images=np.concatenate(imgs), labels=np.concatenate(labs), phase=np.concatenate(phase),
             instruction=np.array(instr))
    print(f"wrote {out}: {len(instr)} frames from {len(paths)} held-out episodes")


if __name__ == "__main__":
    main()
