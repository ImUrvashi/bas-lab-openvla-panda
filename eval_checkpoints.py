"""
Offline check of LoRA checkpoints on held-out frames: predicted 7-D action vs the expert label.

    python panda_tests/eval_checkpoints.py --frames val_frames.npz --adapters A1/lora_adapter A2/lora_adapter ...

Loads the 4-bit base once and swaps adapters. Reports per checkpoint: mean cosine of the xyz move
(overall, lateral phases REACH/TRANSFER, vertical phases), share of frames with cosine > 0.7, gripper agreement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor

import finetune_panda as F
import policy_server as S

LATERAL = {"REACH", "TRANSFER"}
VERTICAL = {"DESCEND", "LIFT", "LOWER", "RETREAT"}


def score(pred, labels, phase):
    moving = np.linalg.norm(labels[:, :3], axis=1) > 1e-3
    cos = np.sum(pred[:, :3] * labels[:, :3], 1) / (
        np.linalg.norm(pred[:, :3], axis=1) * np.linalg.norm(labels[:, :3], axis=1) + 1e-9)
    grip = np.mean((pred[:, 6] > 0.5) == (labels[:, 6] > 0.5))
    pick = lambda names: moving & np.isin(phase, list(names))
    return {"cos": float(cos[moving].mean()), "cos_lateral": float(cos[pick(LATERAL)].mean()),
            "cos_vertical": float(cos[pick(VERTICAL)].mean()), "frac_cos_gt_0.7": float((cos[moving] > 0.7).mean()),
            "gripper_agree": float(grip)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.frames, allow_pickle=True)
    instr = str(d["instruction"])
    proc = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    base = prepare_model_for_kbit_training(F.load_vla(args.vla_path, quantize=True), use_gradient_checkpointing=False)
    model, results = None, {}
    for k, path in enumerate(args.adapters):
        name = f"a{k}"
        if model is None:
            model = PeftModel.from_pretrained(base, str(path), adapter_name=name)
        else:
            model.load_adapter(str(path), adapter_name=name)
        model.set_adapter(name)
        model.eval()
        inner = model.base_model.model
        inner.norm_stats = json.loads((path.parent / "dataset_statistics.json").read_text())
        S.STATE.update(proc=proc, inner=inner, unnorm_key=F.P.DATASET_NAME)
        pred = np.stack([S.predict(instr, img) for img in d["images"]])
        results[str(path)] = score(pred, d["labels"], d["phase"])
        print(json.dumps({"adapter": str(path), **{m: round(v, 3) for m, v in results[str(path)].items()}}), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
