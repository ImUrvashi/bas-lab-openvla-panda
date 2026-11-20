"""Convert a 7D end-effector policy action into bounded Panda joint targets.

This is a one-request-at-a-time bridge for the Unreal prototype, not a 10 Hz
controller. The default sample mode uses a fixed 3 cm hand delta and does not
load OpenVLA. Later, --upstream-url can forward observations to policy_server.py.

    .venv/bin/python policy_ik_gateway.py --port 8000
    .venv/bin/python policy_ik_gateway.py --port 8000 --upstream-url http://127.0.0.1:8001/act
"""

import argparse
import base64
import binascii
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from openvla_adapter import PandaKinematics, decode_action


IMAGE_SHAPE = [224, 224, 3]
MAX_BODY_BYTES = 1_000_000
MAX_HAND_STEP_M = 0.03
MAX_JOINT_STEP_RAD = 0.15
MAX_IK_RESIDUAL_M = 0.01
SAMPLE_ACTION = [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def finite_vector(value, length, name):
    if (not isinstance(value, list) or len(value) != length or
            any(not isinstance(x, (int, float)) or isinstance(x, bool) or
                not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return np.asarray(value, dtype=np.float64)


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if not isinstance(request.get("instruction"), str) or not request["instruction"].strip():
        raise ValueError("instruction must be nonempty text")
    if request.get("shape") != IMAGE_SHAPE:
        raise ValueError("shape must be [224, 224, 3]")
    try:
        rgb = base64.b64decode(request["image_b64"], validate=True)
    except (KeyError, TypeError, binascii.Error) as exc:
        raise ValueError("image_b64 must contain base64 RGB bytes") from exc
    if len(rgb) != 224 * 224 * 3:
        raise ValueError("image_b64 must contain 224x224 raw RGB bytes")
    joints = finite_vector(request.get("joints_rad"), 7, "joints_rad")
    opening = request.get("gripper_opening")
    if (not isinstance(opening, (int, float)) or isinstance(opening, bool) or
            not math.isfinite(opening) or not 0 <= opening <= 1):
        raise ValueError("gripper_opening must be a number from 0 to 1")
    return joints, float(opening)


def convert_action(kin, joints, opening, action):
    action = finite_vector(action, 7, "action").copy()
    distance = np.linalg.norm(action[:3])
    if distance > MAX_HAND_STEP_M:
        action[:3] *= MAX_HAND_STEP_M / distance
    action[6] = np.clip(action[6], 0.0, 1.0)
    target8, residual = decode_action(kin, np.r_[joints, opening], action)
    if not math.isfinite(residual) or residual > MAX_IK_RESIDUAL_M:
        raise ValueError(f"IK residual too large: {residual:.4f} m")
    target8 = np.asarray(target8, dtype=np.float64)
    target8[:7] = np.clip(target8[:7], joints - MAX_JOINT_STEP_RAD, joints + MAX_JOINT_STEP_RAD)
    target8[:7] = np.clip(target8[:7], kin.q_lo, kin.q_hi)
    return action.tolist(), target8.tolist(), residual


class GatewayHandler(BaseHTTPRequestHandler):
    kin = None
    upstream_url = None
    requests = 0
    lock = threading.Lock()

    def send_json(self, code, value):
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self.send_json(200, {"ok": True, "mode": "upstream" if self.upstream_url else "sample",
                                         "requests": self.requests})
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/act":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            joints, opening = validate_request(request)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)})

        start = time.monotonic()
        try:
            if self.upstream_url:
                upstream_request = urllib.request.Request(
                    self.upstream_url,
                    json.dumps({key: request[key] for key in ("instruction", "image_b64", "shape")}).encode(),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(upstream_request, timeout=30) as response:
                    action = json.load(response)["action"]
            else:
                action = SAMPLE_ACTION
            with self.lock:
                safe_action, targets, residual = convert_action(self.kin, joints, opening, action)
                type(self).requests += 1
            print(f"request {self.requests}: IK residual {residual:.6f} m", flush=True)
            self.send_json(200, {"action": safe_action, "joint_targets": targets,
                                 "ik_residual_m": residual,
                                 "latency_s": round(time.monotonic() - start, 3)})
        except (ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
            self.send_json(502, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", help="OpenVLA POST /act endpoint; omit for fixed sample action")
    args = parser.parse_args()
    GatewayHandler.kin = PandaKinematics()
    GatewayHandler.upstream_url = args.upstream_url
    server = ThreadingHTTPServer(("127.0.0.1", args.port), GatewayHandler)
    print(f"IK_GATEWAY_READY http://127.0.0.1:{args.port} mode="
          f"{'upstream' if args.upstream_url else 'sample'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
