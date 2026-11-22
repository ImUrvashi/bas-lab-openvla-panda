"""
Offline check of LoRA checkpoints on held-out frames: predicted 7-D action vs the expert label.

    python panda_tests/eval_checkpoints.py --frames val_frames.npz --adapters A1/lora_adapter A2/lora_adapter ...

Loads the 4-bit base once and swaps adapters. Reports per checkpoint: mean cosine of the xyz move
(overall, lateral and vertical phases), share of frames with cosine > 0.7, gripper agreement; per instruction
when the frames hold several. Valve frames (TURN) also report the predicted yaw rate against the label's.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor

import finetune_panda as F
import policy_server as S

LATERAL = {"REACH", "TRANSFER", "H_REACH", "PULL", "PUSH"}
VERTICAL = {"DESCEND", "LIFT", "LOWER", "RETREAT", "H_DESCEND", "RISE"}


def score(pred, labels, phase):
    moving = np.linalg.norm(labels[:, :3], axis=1) > 1e-3
    cos = np.sum(pred[:, :3] * labels[:, :3], 1) / (
        np.linalg.norm(pred[:, :3], axis=1) * np.linalg.norm(labels[:, :3], axis=1) + 1e-9)
    grip = np.mean((pred[:, 6] > 0.5) == (labels[:, 6] > 0.5))
    pick = lambda names: moving & np.isin(phase, list(names))
    mean = lambda m: float(cos[m].mean()) if m.any() else float("nan")
    out = {"cos": mean(moving), "cos_lateral": mean(pick(LATERAL)), "cos_vertical": mean(pick(VERTICAL)),
           "frac_cos_gt_0.7": float((cos[moving] > 0.7).mean()) if moving.any() else float("nan"),
           "gripper_agree": float(grip)}
    turn = phase == "TURN"
    if turn.any():
        out["turn_yaw_ratio"] = float(pred[turn, 5].mean() / (labels[turn, 5].mean() + 1e-9))
    return out


def score_all(pred, labels, phase, instr):
    res = {"all": score(pred, labels, phase)}
    for text in sorted(set(instr)) if len(set(instr)) > 1 else []:
        m = instr == text
        res[text] = score(pred[m], labels[m], phase[m])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.frames, allow_pickle=True)
    instr = np.broadcast_to(np.asarray(d["instruction"]).astype(str), (len(d["images"]),))
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
        pred = np.stack([S.predict(text, img) for text, img in zip(instr, d["images"])])
        results[str(path)] = score_all(pred, d["labels"], d["phase"], instr)
        for group, r in results[str(path)].items():
            print(json.dumps({"adapter": str(path), "group": group[:24], **{m: round(v, 3) for m, v in r.items()}}),
                  flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
