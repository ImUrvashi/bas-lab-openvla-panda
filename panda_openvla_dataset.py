"""
OpenVLA training-dataset adapter for the Panda pick-and-place pilot data.

Pipeline
    demonstrations/episode_*.npz
        --(openvla_adapter.encode_action)-->  7-D EE-delta actions
        --(q01/q99 bounds normalisation)-->   actions in [-1, 1]  (gripper left as-is)
        --(resize 240x320 -> 224x224)-->      images
    saved to  panda_openvla_data/  (portable; copy this folder to the training machine)

    PandaOpenVLADataset(Dataset) then yields exactly what prismatic's RLDSBatchTransform
    yields: {pixel_values, input_ids, labels, dataset_name}, so it works with
    PaddedCollatorForActionPrediction and the stock OpenVLA training loop.

Conventions (must be identical at inference time -- use the helpers below):
    format_instruction()   : instruction text -> lowercase, no trailing punctuation
    resize_image()         : same 224x224 squash-resize used for training
    unnormalize_actions()  : inverse of normalize_actions(), same as OpenVLA predict_action
    openvla_adapter.decode_action() : 7-D delta -> 8-D joint target for PandaActionInterface

Nothing here modifies demonstrations/ or lerobot_pilot/.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

try:  # torch is only needed for the Dataset itself, not for build()
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:  # pragma: no cover
    _DatasetBase = object

PROJECT_DIR = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_DIR / "demonstrations"
OUT_DIR = PROJECT_DIR / "panda_openvla_data"

DATASET_NAME = "panda_pilot"
IMAGE_SIZE = (224, 224)  # OpenVLA-7B vision backbone input (config.image_sizes)
MIN_RANGE = 0.01         # widen q01/q99 of nearly-constant dims to at least this range
MASK = [True] * 6 + [False]  # gripper (already 0..1) is not rescaled, as in OpenVLA/Bridge
IGNORE_INDEX = -100
ACTION_DIM = 7


# --------------------------------------------------------------------------- helpers
def format_instruction(text):
    """Same text transform for training and inference prompts."""
    return str(text).strip().rstrip(".!?").strip().lower()


def resize_image(img_uint8):
    """(H, W, 3) uint8 -> (224, 224, 3) uint8. Plain squash-resize, no crop (as OpenVLA's RLDS)."""
    return np.asarray(Image.fromarray(img_uint8).resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.uint8)


def compute_bounds(actions7, min_range=MIN_RANGE):
    """Pooled stats in OpenVLA's dataset_statistics format ('action' block)."""
    a = np.asarray(actions7, dtype=np.float64)
    q01, q99 = np.quantile(a, 0.01, axis=0), np.quantile(a, 0.99, axis=0)
    # Nearly-constant dims would be stretched to the full [-1, 1] range (pure noise).
    # Widen their bounds symmetrically so normalised values stay small.
    for i in range(ACTION_DIM):
        if MASK[i] and (q99[i] - q01[i]) < min_range:
            mid = 0.5 * (q99[i] + q01[i])
            q01[i], q99[i] = mid - 0.5 * min_range, mid + 0.5 * min_range
    return {
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "max": a.max(0).tolist(),
        "min": a.min(0).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": MASK,
    }


def normalize_actions(a7, stats):
    """BOUNDS_Q99 normalisation used by OpenVLA's RLDS pipeline."""
    a7 = np.asarray(a7, dtype=np.float64)
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    norm = np.clip(2.0 * (a7 - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)
    return np.where(np.asarray(stats["mask"]), norm, a7).astype(np.float32)


def unnormalize_actions(norm7, stats):
    """Inverse of normalize_actions(); identical to OpenVLA predict_action's un-normalisation."""
    norm7 = np.asarray(norm7, dtype=np.float64)
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    return np.where(np.asarray(stats["mask"]), 0.5 * (norm7 + 1.0) * (hi - lo) + lo, norm7)


# --------------------------------------------------------------------------- build / export
IDLE_POS_M = 3e-4   # a step is "idle" if the hand moves < 0.3 mm ...
IDLE_ROT = 5e-4     # ... and rotates < 0.0005 rad ...


def load_split_map(demo_dir):
    """file name -> 'train'/'val' from batch_manifest_seed*.json; unlisted episodes are 'train'."""
    split = {}
    for mp in sorted(Path(demo_dir).glob("batch_manifest_seed*.json")):
        for e in json.loads(mp.read_text()).get("episodes", []):
            split[e["file"]] = e.get("split", "train")
    return split


def lead_gripper(g, k_close, k_open):
    """
    Move each gripper command k frames earlier (labels only; motion labels are untouched).

    A demo reaches the grasp pose with a decelerating tail of near-identical frames and only then flips
    the gripper command, so one image carries both labels. Labelling the whole tail with the upcoming
    command makes the rule visual: "hand at the object -> close" / "object at the target -> open".
    """
    g_out = g.copy()
    for i in range(1, len(g)):
        if g[i] < g[i - 1] - 0.5 and k_close > 0:      # open -> close command
            g_out[max(0, i - k_close):i] = g[i]
        elif g[i] > g[i - 1] + 0.5 and k_open > 0:     # close -> open command
            g_out[max(0, i - k_open):i] = g[i]
    return g_out


def relabel_onsets(a7, events, thr=8e-4, settle_close=1, settle_open=2):
    """
    Give near-stationary frames that precede a motion the label of that motion's first substantive step.

    Every motion segment starts from rest with an ease-in, so its first frames carry labels of only
    0.1-0.5 mm: a policy fed an unchanged image repeats ~0 forever and never starts moving. Frames that
    are still (< thr metres of hand motion) take the action of the next frame that moves (>= thr), except:
      * frames before a gripper command (`events`) keep their labels: the arm must hold still while the
        gripper closes/opens,
      * the frames after a command keep theirs for `settle_close` / `settle_open` frames (the command frame
        itself still shows open fingers; one frame later a closed gripper is already visible, so after a
        close only the command frame is protected, after a release two).
    The gripper entry is untouched; frames with no later motion are unchanged.
    Returns (new_a7, n_relabeled).
    """
    a = a7.copy()
    n = len(a)
    mag = np.linalg.norm(a[:, :3], axis=1)
    moving = mag >= thr
    ev = sorted((int(e), kind) for e, kind in events)  # (frame, "close"|"open")
    nxt = np.full(n, -1, dtype=int)  # next moving frame at or after t
    last = -1
    for t in range(n - 1, -1, -1):
        if moving[t]:
            last = t
        nxt[t] = last
    count = 0
    for t in range(n):
        if moving[t] or nxt[t] < 0:
            continue
        m = nxt[t]
        if any(t < e <= m for e, _ in ev):       # a gripper command comes first: hold still for it
            continue
        if any(0 <= t - e < (settle_close if kind == "close" else settle_open) for e, kind in ev):
            continue
        a[t, :6] = a7[m, :6]
        count += 1
    return a, count


def build(demo_dir=DEMO_DIR, out_dir=OUT_DIR, min_episode=0, lead_close=0, lead_open=0,
          onset_relabel=False, stats_from=None, images_ref=None):
    import openvla_adapter as A  # needs mujoco; only required for building, not for training

    paths = sorted(Path(demo_dir).glob("episode_*.npz"))
    paths = [p for p in paths if int(p.stem.split("_")[-1]) >= min_episode]
    if not paths:
        raise FileNotFoundError(f"No episodes (>= {min_episode}) in {demo_dir}")
    kin = A.PandaKinematics()
    split_map = load_split_map(demo_dir)

    # Pass 1: sizes only (lazy npz access), so that images can be preallocated.
    lengths = [int(np.load(p, allow_pickle=True)["states"].shape[0]) for p in paths]
    total = int(sum(lengths))
    images = None if images_ref else np.empty((total, *IMAGE_SIZE, 3), dtype=np.uint8)
    a7 = np.empty((total, ACTION_DIM), dtype=np.float32)
    ep_idx = np.empty(total, dtype=np.int32)
    frame_idx = np.empty(total, dtype=np.int32)
    instr = []

    ep_info = []
    relabeled = 0
    pos = 0
    for e, (p, n) in enumerate(zip(paths, lengths)):
        ep = A.convert_episode(kin, p)
        assert len(ep["actions7"]) == n
        # gripper command flips in the ORIGINAL labels (before any lead shift): hold-still boundaries
        events = [(i, "close" if ep["actions7"][i, 6] < ep["actions7"][i - 1, 6] else "open")
                  for i in range(1, n) if abs(ep["actions7"][i, 6] - ep["actions7"][i - 1, 6]) > 0.5]
        if lead_close or lead_open:
            ep["actions7"][:, 6] = lead_gripper(ep["actions7"][:, 6], lead_close, lead_open)
        if onset_relabel:
            ep["actions7"], k = relabel_onsets(ep["actions7"], events)
            relabeled += k
        if images is not None:
            for j, im in enumerate(ep["images"]):
                images[pos + j] = resize_image(im)
        a7[pos:pos + n] = ep["actions7"]
        ep_idx[pos:pos + n] = e
        frame_idx[pos:pos + n] = np.arange(n)
        instr += [ep["instruction"]] * n
        ep_info.append({"episode": e, "file": p.name, "frames": n, "instruction": ep["instruction"],
                        "split": split_map.get(p.name, "train")})
        pos += n
        if (e + 1) % 20 == 0:
            print(f"  converted {e + 1}/{len(paths)} episodes", flush=True)

    train_eps = [i["episode"] for i in ep_info if i["split"] == "train"]
    val_eps = [i["episode"] for i in ep_info if i["split"] == "val"]
    train_mask = np.isin(ep_idx, train_eps)

    # Normalisation statistics come from TRAIN episodes only (no leakage from validation).
    if stats_from is not None:  # keep the token <-> action mapping of an existing model
        stats = json.loads(Path(stats_from).read_text())[DATASET_NAME]["action"]
        print(f"  using normalisation statistics from {stats_from}", flush=True)
    else:
        stats = compute_bounds(a7[train_mask])
    if onset_relabel:
        print(f"  onset relabel: {relabeled} of {total} frames relabeled "
              f"({100.0 * relabeled / total:.1f}%)", flush=True)
    a_norm = normalize_actions(a7, stats)

    # Frames where nothing happens (hold phases): kept in the data, flagged for optional subsampling.
    prev_grip = np.concatenate([[a7[0, 6]], a7[:-1, 6]])
    new_ep = np.concatenate([[True], ep_idx[1:] != ep_idx[:-1]])
    grip_change = (~new_ep) & (np.abs(a7[:, 6] - prev_grip) > 1e-6)
    idle = (np.linalg.norm(a7[:, :3], axis=1) < IDLE_POS_M) & (np.abs(a7[:, 3:6]).max(1) < IDLE_ROT)
    idle &= ~grip_change

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    arrays = dict(actions7=a7, actions_norm=a_norm, episode_index=ep_idx, frame_index=frame_idx, idle=idle)
    if images is not None:
        arrays["images"] = images
    np.savez_compressed(out_dir / "data.npz", **arrays)
    if images_ref:
        (out_dir / "images_ref.txt").write_text(str(images_ref))
    (out_dir / "instructions.json").write_text(json.dumps(instr))
    (out_dir / "split.json").write_text(json.dumps(
        {"train": train_eps, "val": val_eps, "episodes": ep_info,
         "build": {"min_episode": min_episode, "gripper_lead_close": lead_close,
                   "gripper_lead_open": lead_open, "onset_relabel": onset_relabel,
                   "stats_from": str(stats_from) if stats_from else None,
                   "images_ref": images_ref}}, indent=1))
    # Same layout as OpenVLA's dataset_statistics.json (needed at inference as norm_stats).
    stats_doc = {
        DATASET_NAME: {
            "action": stats,
            "num_transitions": int(train_mask.sum()),
            "num_trajectories": len(train_eps),
        }
    }
    (out_dir / "dataset_statistics.json").write_text(json.dumps(stats_doc, indent=2))
    return out_dir, stats


# --------------------------------------------------------------------------- Dataset
class _ImageParts:
    """Read-only concatenation of several memory-mapped image arrays, indexed like one array."""

    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(x) for x in parts])
        self.shape = (int(self.offsets[-1]),) + tuple(parts[0].shape[1:])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, j):
        k = int(np.searchsorted(self.offsets, j, side="right")) - 1
        return self.parts[k][int(j - self.offsets[k])]


class PandaOpenVLADataset(_DatasetBase):
    """
    Map-style dataset matching prismatic's RLDSBatchTransform output.

    Pass processor.tokenizer, ActionTokenizer(processor.tokenizer),
    processor.image_processor.apply_transform and PurePromptBuilder, as finetune.py does.
    """

    def __init__(
        self,
        data_dir,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn,
        episodes=None,
        predict_stop_token=True,
        idle_keep_frac=1.0,
        seed=0,
    ):
        data_dir = Path(data_dir)
        d = np.load(data_dir / "data.npz")
        # An uncompressed images.npy next to data.npz is memory-mapped (low RAM; used on small machines).
        npy = data_dir / "images.npy"
        ref = data_dir / "images_ref.txt"
        parts_json = data_dir / "images_parts.json"
        if ref.exists():  # images stored once in another export (same episodes, same order)
            npy = (data_dir / ref.read_text().strip()).resolve()
        if parts_json.exists():  # several image files concatenated in order
            arrs = []
            for part in json.loads(parts_json.read_text()):
                a = np.load((data_dir / part["path"]).resolve(), mmap_mode="r")
                assert len(a) == part["n"], f"{part['path']}: {len(a)} frames, expected {part['n']}"
                arrs.append(a)
            self.images = _ImageParts(arrs)
        else:
            self.images = np.load(npy, mmap_mode="r") if npy.exists() else d["images"]
        self.actions_norm = d["actions_norm"]
        self.episode_index = d["episode_index"]
        self.instructions = json.loads((data_dir / "instructions.json").read_text())
        self.dataset_statistics = json.loads((data_dir / "dataset_statistics.json").read_text())

        keep = np.ones(len(self.images), dtype=bool)
        if episodes is not None:
            keep &= np.isin(self.episode_index, list(episodes))
        if idle_keep_frac < 1.0 and "idle" in d.files:
            # Hold-phase frames (no motion, no gripper change) are over-represented; keep a fraction.
            rng = np.random.default_rng(seed)
            keep &= (~d["idle"]) | (rng.random(len(keep)) < idle_keep_frac)
        self.indices = np.where(keep)[0]

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import torch

        j = int(self.indices[i])
        action = self.actions_norm[j]
        lang = format_instruction(self.instructions[j])
        img = Image.fromarray(np.ascontiguousarray(self.images[j]))

        prompt_builder = self.prompt_builder_fn("openvla")
        for turn, value in (
            ("human", f"What action should the robot take to {lang}?"),
            ("gpt", self.action_tokenizer(action)),
        ):
            prompt_builder.add_turn(turn, value)

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Loss only on the action tokens (+ stop token), as in RLDSBatchTransform.
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return dict(
            pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=DATASET_NAME
        )


# --------------------------------------------------------------------------- self-check
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--min_episode", type=int, default=0, help="only episode_NNN.npz with NNN >= this")
    ap.add_argument("--gripper_lead_close", type=int, default=0, help="frames to advance the close command")
    ap.add_argument("--gripper_lead_open", type=int, default=0, help="frames to advance the open command")
    ap.add_argument("--onset_relabel", action="store_true", help="relabel still frames that precede a motion")
    ap.add_argument("--stats_from", type=Path, default=None, help="reuse these dataset_statistics.json bounds")
    ap.add_argument("--images_ref", default=None,
                    help="do not store images; reference an existing images.npy (path relative to out_dir)")
    args = ap.parse_args()
    if (args.out_dir / "data.npz").exists():
        raise FileExistsError(f"{args.out_dir} already holds an export; choose a new --out_dir.")
    out_dir, stats = build(out_dir=args.out_dir, min_episode=args.min_episode,
                           lead_close=args.gripper_lead_close, lead_open=args.gripper_lead_open,
                           onset_relabel=args.onset_relabel, stats_from=args.stats_from,
                           images_ref=args.images_ref)
    d = np.load(out_dir / "data.npz")
    sp = json.loads((out_dir / "split.json").read_text())
    print(f"split: {len(sp['train'])} train episodes, {len(sp['val'])} val episodes;"
          f" idle frames: {100.0 * d['idle'].mean():.1f}%")
    n = len(d["actions7"])
    print(f"Wrote {out_dir.name}/  ({n} samples, {len(set(d['episode_index'].tolist()))} episodes)")
    if "images" in d.files:
        print(f"  images {d['images'].shape} {d['images'].dtype}")
    print(f"  actions_norm range [{d['actions_norm'].min():.3f}, {d['actions_norm'].max():.3f}]")

    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    print("\n  dim      q01       q99     norm-mean  norm-std  %clipped")
    for i, nm in enumerate(names):
        col = d["actions_norm"][:, i]
        clipped = 100.0 * np.mean(np.abs(col) >= 0.9999) if MASK[i] else 0.0
        print(f"  {nm:7s} {stats['q01'][i]:8.4f} {stats['q99'][i]:9.4f} "
              f"{col.mean():10.3f} {col.std():9.3f} {clipped:8.1f}")

    # normalise -> unnormalise round trip (only clipped tails may differ)
    back = unnormalize_actions(d["actions_norm"], stats)
    err = np.abs(back - d["actions7"])
    inside = np.ones(len(back), dtype=bool)
    print(f"\n  normalise/unnormalise max abs error: {err.max():.2e}"
          f" (clipped tail samples excluded would be ~0)")

    # quantisation error from 256 bins: half a bin width in normalised units
    bin_w = 2.0 / 255.0
    q_err = bin_w / 2.0 * 0.5 * (np.asarray(stats["q99"]) - np.asarray(stats["q01"]))
    print("  expected 256-bin quantisation error per dim (metres/rad):")
    print("   ", ", ".join(f"{nm}={e:.5f}" for nm, e in zip(names[:6], q_err[:6])))
    print("\nPortable output folder: panda_openvla_data/  (data.npz, instructions.json,"
          " dataset_statistics.json)")


if __name__ == "__main__":
    main()
