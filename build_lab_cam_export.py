"""
Lab-only training export with the closer side camera (D67/D72).

Replays the stored lab expert episodes (demonstrations_bas/expert_*.npz) by their executed actions in the
lab scene (replay matches the stored hand/cube to 0.00 mm), renders every kept frame with the side camera and
writes an export in the same format as build_combined_export.py (single image part). Labels are exactly the v11
labels: soft-relabelled expert labels, same stride, same normalisation statistics (copied from --stats_from).

    python build_lab_cam_export.py --stats_from panda_openvla_data_v11 --out panda_openvla_data_v12_side
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import closed_loop_panda as C
import expert_policy as E
import panda_openvla_dataset as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_dir", type=Path, default=Path("demonstrations_bas"))
    ap.add_argument("--stats_from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    stats_doc = json.loads((args.stats_from / "dataset_statistics.json").read_text())
    stats = stats_doc[P.DATASET_NAME]["action"]
    env = C.PandaEnv(C.BAS_SCENE, camera="side")
    paths = sorted(args.expert_dir.glob("expert_*.npz"))[:args.limit]
    imgs, a7, ep_idx, frm_idx, instr, info = [], [], [], [], [], []
    worst_h = worst_c = 0.0
    t0 = time.time()
    for k, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        ex, hand0, cube0 = d["executed7"], d["hand"], d["cube"]
        lab = E.relabel_soft(hand0, cube0, d["phase"], d["labels7"])
        keep = set(range(0, len(lab), args.stride))
        env.reset(d["start_xy"], d["target_xy"])
        frames = []
        for t in range(len(ex)):
            worst_h = max(worst_h, float(np.linalg.norm(env.hand() - hand0[t])))
            worst_c = max(worst_c, float(np.linalg.norm(env.cube() - cube0[t])))
            if t in keep:
                frames.append(P.resize_image(env.image()))
            env.execute_delta(ex[t])
        idx = np.array(sorted(keep))
        imgs.append(np.stack(frames)); a7.append(lab[idx])
        ep_idx += [k] * len(idx); frm_idx += idx.tolist(); instr += [str(d["instruction"])] * len(idx)
        info.append({"episode": k, "file": p.name, "frames": int(len(idx)), "instruction": str(d["instruction"]),
                     "split": "train", "source": "expert", "sigma_m": float(d["sigma"])})
        if k % 10 == 0:
            print(f"{k + 1}/{len(paths)} {p.name} elapsed {time.time() - t0:.0f}s replay err hand {1000 * worst_h:.2f} mm cube {1000 * worst_c:.2f} mm", flush=True)
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
         "build": {"camera": "side", "side_camera": C.SIDE_CAMERA, "stride": args.stride, "soft_relabel": True,
                   "stats_from": str(args.stats_from)}}, indent=1))
    (args.out / "dataset_statistics.json").write_text(json.dumps(stats_doc, indent=2))
    print(f"wrote {args.out}: {len(a7)} frames, {len(info)} episodes, images {imgs.shape} = {imgs.nbytes / 2**30:.2f} GiB; "
          f"max replay error hand {1000 * worst_h:.2f} mm, cube {1000 * worst_c:.2f} mm; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
