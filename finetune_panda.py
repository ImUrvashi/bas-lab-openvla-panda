"""
finetune_panda.py -- single-GPU (4-bit) LoRA fine-tuning of OpenVLA-7B on the Panda pilot dataset.

Adapted from OpenVLA's vla-scripts/finetune.py, with these deliberate differences:
  * data comes from PandaOpenVLADataset (map-style, epoch loop) instead of the RLDS pipeline
  * single GPU, no DDP; no Weights & Biases (metrics go to stdout + metrics.jsonl)
  * workaround for accelerate>=1.x + transformers 4.40.1 (`.to()` on a bnb model) -- see below
  * only the small LoRA adapter is saved by default (no merge into the 7B base)
  * writes everything under --run_root_dir; never touches anything outside it

Run (from the OpenVLA repo root, inside its .venv):
    CUDA_VISIBLE_DEVICES=0 python panda_tests/finetune_panda.py \
        --data_dir panda_tests/panda_openvla_data --run_root_dir panda_tests/runs --max_steps 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import transformers.modeling_utils as _mu
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent  # OpenVLA repo root (panda_tests/ lives directly inside it)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))  # so `prismatic` imports without `pip install -e .`
import panda_openvla_dataset as P


def _skip_vla_package_init():
    """prismatic/vla/__init__.py imports the RLDS (TensorFlow) data pipeline, which neither fine-tuning on
    our own dataset nor inference needs. `import prismatic` itself pulls in prismatic.vla, so this must run
    BEFORE any prismatic import: it registers the package without executing its __init__, which lets
    `prismatic.vla.action_tokenizer` import cleanly without TensorFlow/dlimp installed."""
    import types

    if "prismatic.vla" not in sys.modules and (_REPO / "prismatic" / "vla").is_dir():
        stub = types.ModuleType("prismatic.vla")
        stub.__path__ = [str(_REPO / "prismatic" / "vla")]
        sys.modules["prismatic.vla"] = stub


_skip_vla_package_init()
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer


# --- Workaround: accelerate>=1.x calls model.to(device) at the end of dispatch_model, which
# transformers 4.40.1 forbids for bitsandbytes models. Weights are already on the GPU after
# quantized loading, so skip it and move leftover CPU buffers (e.g. rotary inv_freq) by hand.
_orig_dispatch = _mu.dispatch_model


def _dispatch_skip_bnb(model, device_map=None, **kw):
    if getattr(model, "quantization_method", None) is not None and getattr(model, "is_quantized", False):
        model.hf_device_map = dict(device_map)
        for m in model.modules():
            for n, b in m._buffers.items():
                if b is not None and b.device.type == "cpu":
                    m._buffers[n] = b.to("cuda:0")
        return model
    return _orig_dispatch(model, device_map=device_map, **kw)


_mu.dispatch_model = _dispatch_skip_bnb


def load_vla(vla_path, quantize):
    qcfg = None
    if quantize:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    return AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=qcfg,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0},
    )


def gib(x):
    return round(x / 2**30, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--data_dir", type=Path, default=Path("panda_tests/panda_openvla_data"))
    ap.add_argument("--run_root_dir", type=Path, default=Path("panda_tests/runs"))
    ap.add_argument("--run_id_note", default="dryrun")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=20, help="optimizer steps")
    ap.add_argument("--save_steps", type=int, default=0, help="0 = save only at the end")
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--no_quantization", action="store_true", help="bf16 base instead of 4-bit")
    ap.add_argument("--train_episodes", type=int, nargs="*", default=None,
                    help="episode ids; default: the 'train' list of <data_dir>/split.json, else all")
    ap.add_argument("--idle_keep_frac", type=float, default=1.0,
                    help="fraction of idle (no-motion) frames to keep; 1.0 keeps all")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verify_reload", action="store_true", help="reload the saved adapter and compare outputs")
    ap.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                    help="cosine = linear warmup, then cosine decay to min_lr_frac * learning_rate at max_steps")
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--keep_checkpoints", action="store_true",
                    help="keep every --save_steps checkpoint in checkpoints/step_NNNNN/ instead of overwriting")
    ap.add_argument("--resume_adapter", type=Path, default=None,
                    help="LoRA adapter directory to continue training from (weights only; fresh optimizer)")
    args = ap.parse_args()

    if sys.platform == "win32":
        # Ask Windows not to sleep while THIS process runs (a per-process request that vanishes when the
        # process exits; no system setting is changed). The display may still turn off.
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    assert torch.cuda.is_available(), "needs a GPU"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    exp_id = (
        f"{args.vla_path.split('/')[-1]}+panda_pilot+b{args.batch_size * args.grad_accumulation_steps}"
        f"+lr-{args.learning_rate}+lora-r{args.lora_rank}"
        f"{'' if args.no_quantization else '+q-4bit'}--{args.run_id_note}"
    )
    run_dir = args.run_root_dir / exp_id
    adapter_dir = run_dir / "lora_adapter"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}", flush=True)
    (run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    t0 = time.time()
    vla = load_vla(args.vla_path, quantize=not args.no_quantization)
    print(f"model loaded in {time.time() - t0:.1f}s | GPU allocated {gib(torch.cuda.memory_allocated(0))} GiB", flush=True)

    if not args.no_quantization:
        vla = prepare_model_for_kbit_training(vla)  # as in the stock script (casts non-quantized params to fp32)
    if args.resume_adapter is not None:
        vla = PeftModel.from_pretrained(vla, str(args.resume_adapter), is_trainable=True)
        print(f"resumed LoRA weights from {args.resume_adapter}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_cfg)
    vla.print_trainable_parameters()
    print(f"after LoRA wrap: GPU allocated {gib(torch.cuda.memory_allocated(0))} GiB", flush=True)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    train_eps = args.train_episodes
    split_path = args.data_dir / "split.json"
    if train_eps is None and split_path.exists():
        train_eps = json.loads(split_path.read_text())["train"]
        print(f"using the 'train' split of {split_path.name}: {len(train_eps)} episodes", flush=True)
    dataset = P.PandaOpenVLADataset(
        args.data_dir,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder,
        episodes=train_eps,
        idle_keep_frac=args.idle_keep_frac,
        seed=args.seed,
    )
    print(f"training samples: {len(dataset)}", flush=True)
    # dataset_statistics.json is required at inference time to de-normalize actions.
    (run_dir / "dataset_statistics.json").write_text(json.dumps(dataset.dataset_statistics, indent=2))

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=collator, num_workers=args.num_workers)

    trainable = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)

    import math

    def lr_factor(step):  # `step` = optimizer steps already taken
        if args.lr_schedule == "constant":
            return 1.0
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    metrics_path = run_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "w")

    def save_checkpoint(tag, step=None):
        # save_embedding_layers=False: the (padded) lm_head is not trained, so do not write a 0.5 GB fp32 copy.
        targets = [adapter_dir]
        if args.keep_checkpoints and step is not None:
            ck = run_dir / "checkpoints" / f"step_{step:05d}"
            ck.mkdir(parents=True, exist_ok=True)
            (ck / "dataset_statistics.json").write_text((run_dir / "dataset_statistics.json").read_text())
            targets.append(ck / "lora_adapter")
        for t in targets:
            vla.save_pretrained(t, save_embedding_layers=False)
        processor.save_pretrained(run_dir)
        print(f"saved LoRA adapter ({tag}) -> {targets[-1]}", flush=True)

    vla.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(0)
    micro_step, opt_step, t_start = 0, 0, time.time()
    accum = {"loss": [], "acc": [], "l1": []}
    done = False
    epoch = 0
    while not done:
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"].to(device),
                )
            loss = out.loss
            (loss / args.grad_accumulation_steps).backward()

            # Same logging metrics as the stock script: token accuracy + L1 on de-tokenized actions.
            with torch.no_grad():
                logits = out.logits[:, num_patches:-1]
                preds = logits.argmax(dim=2)
                gt = batch["labels"][:, 1:].to(preds.device)
                mask = gt > action_tokenizer.action_token_begin_idx
                acc = ((preds == gt) & mask).sum().float() / mask.sum().float()
                l1 = torch.nn.functional.l1_loss(
                    torch.tensor(action_tokenizer.decode_token_ids_to_actions(preds[mask].cpu().numpy())),
                    torch.tensor(action_tokenizer.decode_token_ids_to_actions(gt[mask].cpu().numpy())),
                )
            accum["loss"].append(loss.item()); accum["acc"].append(acc.item()); accum["l1"].append(l1.item())
            micro_step += 1

            if micro_step % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                rec = {
                    "step": opt_step, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                    "loss": float(np.mean(accum["loss"])), "action_accuracy": float(np.mean(accum["acc"])),
                    "l1": float(np.mean(accum["l1"])),
                    "peak_gpu_gib": gib(torch.cuda.max_memory_allocated(0)),
                    "elapsed_s": round(time.time() - t_start, 1),
                }
                accum = {"loss": [], "acc": [], "l1": []}
                metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
                print(json.dumps(rec), flush=True)
                if args.save_steps and opt_step % args.save_steps == 0:
                    save_checkpoint(f"step {opt_step}", step=opt_step)
                if opt_step >= args.max_steps:
                    done = True
                    break
        epoch += 1

    metrics_f.close()
    save_checkpoint("final", step=opt_step if (not args.save_steps or opt_step % args.save_steps) else None)
    peak = gib(torch.cuda.max_memory_allocated(0))
    print(f"TRAINING_DONE steps={opt_step} peak_gpu_gib={peak} "
          f"sec_per_step={(time.time() - t_start) / max(opt_step, 1):.2f}", flush=True)

    if args.verify_reload:
        # Reload the adapter onto a fresh base and check it reproduces the trained model's output.
        vla.eval()
        probe = {k: v for k, v in collator([dataset[0]]).items() if k != "dataset_names"}
        probe = {k: (v.to(device) if k != "pixel_values" else v.to(torch.bfloat16).to(device)) for k, v in probe.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = vla(**{k: probe[k] for k in ("input_ids", "attention_mask", "pixel_values")}).logits.float().cpu()
        del vla, optimizer, out, loss, logits, preds
        torch.cuda.empty_cache()
        base = load_vla(args.vla_path, quantize=not args.no_quantization)
        re = PeftModel.from_pretrained(base, adapter_dir)
        re.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            new = re(**{k: probe[k] for k in ("input_ids", "attention_mask", "pixel_values")}).logits.float().cpu()
        diff = (ref - new).abs().max().item()
        print(f"RELOAD_CHECK max|logit diff| = {diff:.3e}", flush=True)
        print("RELOAD_OK" if diff < 1e-1 else "RELOAD_MISMATCH", flush=True)


if __name__ == "__main__":
    main()
