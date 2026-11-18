"""
policy_server.py -- serve a fine-tuned OpenVLA (4-bit base + LoRA adapter) over localhost HTTP.

Runs on the machine with the GPU (copy next to the OpenVLA repo). Binds 127.0.0.1 only; if the
simulator runs on another machine, forward port 8000 to it (e.g. an SSH local port-forward).

    POST /act   {"instruction": str, "image_b64": base64(uint8 HxWx3), "shape": [224,224,3]}
        -> {"action": [dx,dy,dz,droll,dpitch,dyaw,gripper], "latency_s": float}
    GET  /health

Returned actions are already de-normalised (metres / radians / gripper 0..1) using the run's
dataset_statistics.json, i.e. exactly what openvla_adapter.decode_action() expects.
"""

import argparse
import base64
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


def predict(instruction, image_u8):
    proc, inner, key = STATE["proc"], STATE["inner"], STATE["unnorm_key"]
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
            self._send(200, {"ok": True, "adapter": STATE["adapter"], "requests": STATE["n"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/act":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            img = np.frombuffer(base64.b64decode(req["image_b64"]), dtype=np.uint8).reshape(req["shape"])
            t0 = time.time()
            with LOCK:
                action = predict(req["instruction"], img)
                STATE["n"] += 1
            self._send(200, {"action": action.tolist(), "latency_s": round(time.time() - t0, 3)})
        except Exception as e:  # report to the client instead of killing the server
            self._send(500, {"error": repr(e)})

    def log_message(self, fmt, *args):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--adapter_dir", type=Path, required=True, help=".../lora_adapter (run dir holds dataset_statistics.json)")
    ap.add_argument("--stats", type=Path, default=None, help="dataset_statistics.json (default: next to adapter_dir)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    stats_path = args.stats or (args.adapter_dir.parent / "dataset_statistics.json")
    stats = json.loads(stats_path.read_text())
    unnorm_key = P.DATASET_NAME
    assert unnorm_key in stats, f"{unnorm_key} not in {stats_path}"

    print("loading processor + 4-bit base ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    base = F.load_vla(args.vla_path, quantize=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)  # same numerics as training
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.eval()
    inner = model.base_model.model  # OpenVLAForActionPrediction with LoRA layers injected
    inner.norm_stats = stats        # de-normalisation statistics of OUR dataset

    STATE.update(proc=proc, inner=inner, unnorm_key=unnorm_key, adapter=str(args.adapter_dir), n=0)

    # Warm-up (also proves the whole path works before accepting clients).
    t0 = time.time()
    a = predict("pick up the red cube", np.zeros((224, 224, 3), dtype=np.uint8))
    print(f"warm-up action {np.round(a, 4).tolist()} in {time.time() - t0:.2f}s", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"SERVER_READY http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
