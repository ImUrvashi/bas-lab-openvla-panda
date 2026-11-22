"""
policy_server.py -- serve a fine-tuned OpenVLA (4-bit or bf16 base + LoRA adapters) over localhost HTTP.

Runs on the machine with the GPU (copy next to the OpenVLA repo). Binds 127.0.0.1 only; if the
simulator runs on another machine, forward port 8000 to it (e.g. an SSH local port-forward).

    POST /act   {"instruction": str, "image_b64": base64(uint8 HxWx3) or "image_png_b64": base64(PNG), "shape": [224,224,3]}
        -> {"action": [dx,dy,dz,droll,dpitch,dyaw,gripper], "latency_s": float}
    POST /interpret {"text": str}  (with --llm) -> {"task": 0-5, "instruction": str, "reply": str}
    GET  /health

Returned actions are already de-normalised (metres / radians / gripper 0..1) using the run's
dataset_statistics.json, i.e. exactly what openvla_adapter.decode_action() expects.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_panda as F  # load_vla() + accelerate/transformers dispatch workaround
import panda_openvla_dataset as P

LOCK = threading.Lock()
STATE = {}


def build_prompt(instruction):
    # Identical to training: PurePromptBuilder("openvla") -> "In: ...?\nOut:" then action tokens.
    return f"In: What action should the robot take to {P.format_instruction(instruction)}?\nOut:"


def adapter_for(instruction):
    """Name of the LoRA adapter that serves this instruction (first matching --route prefix, else the main one)."""
    text = instruction.strip().lower()
    return next((name for prefix, name in STATE["routes"] if text.startswith(prefix)), "default")


def predict(instruction, image_u8):
    proc, inner, key = STATE["proc"], STATE["inner"], STATE["unnorm_key"]
    name = adapter_for(instruction)
    if name != STATE["active"]:
        STATE["model"].set_adapter(name)
        STATE["active"] = name
    inputs = proc(build_prompt(instruction), Image.fromarray(image_u8)).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        action = inner.predict_action(**inputs, unnorm_key=key, do_sample=False)
    return np.asarray(action, dtype=np.float64)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "adapter": STATE["adapter"], "bf16": STATE["bf16"],
                             "routes": [p for p, _ in STATE["routes"]], "requests": STATE["n"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/interpret":
            return self._interpret()
        if self.path != "/act":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            if "image_png_b64" in req:      # lossless PNG from the gateway (smaller over the VPN)
                img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB"))
            else:
                img = np.frombuffer(base64.b64decode(req["image_b64"]), dtype=np.uint8).reshape(req["shape"])
            t0 = time.time()
            with LOCK:
                action = predict(req["instruction"], img)
                STATE["n"] += 1
            self._send(200, {"action": action.tolist(), "latency_s": round(time.time() - t0, 3)})
        except Exception as e:  # report to the client instead of killing the server
            self._send(500, {"error": repr(e)})

    def _interpret(self):
        if "llm" not in STATE:
            return self._send(503, {"error": "no language front end (start with --llm)"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t0 = time.time()
            with LOCK:
                out = STATE["llm"].interpret(str(req["text"]))
            print(f"interpret {req['text']!r} -> {out}", flush=True)
            self._send(200, {**out, "latency_s": round(time.time() - t0, 3)})
        except Exception as e:
            self._send(500, {"error": repr(e)})

    def log_message(self, fmt, *args):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--adapter_dir", type=Path, required=True, help=".../lora_adapter (run dir holds dataset_statistics.json)")
    ap.add_argument("--stats", type=Path, default=None, help="dataset_statistics.json (default: next to adapter_dir)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bf16", action="store_true", help="bf16 base instead of 4-bit (several times faster, ~16 GB)")
    ap.add_argument("--route", action="append", default=[], metavar="PREFIX=ADAPTER_DIR",
                    help="serve instructions starting with PREFIX with another adapter (same dataset statistics)")
    ap.add_argument("--llm", default=None, help="also load this chat model as the request front end (/interpret)")
    args = ap.parse_args()

    stats_path = args.stats or (args.adapter_dir.parent / "dataset_statistics.json")
    stats = json.loads(stats_path.read_text())
    unnorm_key = P.DATASET_NAME
    assert unnorm_key in stats, f"{unnorm_key} not in {stats_path}"

    print(f"loading processor + {'bf16' if args.bf16 else '4-bit'} base ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    base = F.load_vla(args.vla_path, quantize=not args.bf16)
    if not args.bf16:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)  # same numerics as training
    model = PeftModel.from_pretrained(base, str(args.adapter_dir), adapter_name="default")
    routes = []
    for k, r in enumerate(args.route):
        prefix, path = r.split("=", 1)
        model.load_adapter(path, adapter_name=f"route{k}")
        routes.append((prefix.strip().lower(), f"route{k}"))
        print(f"route {prefix!r} -> {path}", flush=True)
    model.set_adapter("default")
    model.eval()
    inner = model.base_model.model  # OpenVLAForActionPrediction with LoRA layers injected
    inner.norm_stats = stats        # de-normalisation statistics of OUR dataset

    STATE.update(proc=proc, inner=inner, model=model, unnorm_key=unnorm_key, adapter=str(args.adapter_dir), n=0,
                 routes=routes, active="default", bf16=args.bf16)
    if args.llm:
        try:     # the policy must still serve if the language model cannot load
            from llm_frontend import LlamaFrontEnd
            print(f"loading front end {args.llm} ...", flush=True)
            STATE["llm"] = LlamaFrontEnd(args.llm)
            print(f"front end: {STATE['llm'].interpret('please close the coolant valve')}", flush=True)
        except Exception as e:
            STATE.pop("llm", None)
            print(f"front end not loaded: {e!r}", flush=True)

    # Warm-up (also proves the whole path works before accepting clients).
    t0 = time.time()
    a = predict("pick up the red cube", np.zeros((224, 224, 3), dtype=np.uint8))
    print(f"warm-up action {np.round(a, 4).tolist()} in {time.time() - t0:.2f}s", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"SERVER_READY http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
