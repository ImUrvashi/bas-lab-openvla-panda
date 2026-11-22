"""Convert a 7D end-effector policy action into bounded Panda joint targets.

Unreal's bridge calls POST /act once per 0.1 s policy step. Where the 7-D action comes from:
  sample    fixed 3 cm hand delta (connection test)
  expert    expert_policy.py on the true cube/marker state Unreal sends (no model)
  upstream  OpenVLA policy_server.py, e.g. on the GPU server through an SSH tunnel

    .venv/bin/python policy_ik_gateway.py --policy expert
    .venv/bin/python policy_ik_gateway.py --policy expert --record demonstrations_unreal --seed 1
    .venv/bin/python policy_ik_gateway.py --upstream-url http://127.0.0.1:8001/act
"""

import argparse
import base64
import binascii
import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from demo_recorder import DemoRecorder
from expert_policy import ExpertPolicy
from multitask_expert import NOISE as MULTI_NOISE, MultiTaskExpert, TaskState, task_success
from openvla_adapter import PandaKinematics, decode_action
from panda_openvla_dataset import resize_image
from PIL import Image


IMAGE_SHAPES = ([224, 224, 3], [240, 320, 3])  # 320x240 = MuJoCo render, resized here as in training
MAX_BODY_BYTES = 1_000_000
MAX_HAND_STEP_M = 0.03
MAX_JOINT_STEP_RAD = 0.15
MAX_IK_RESIDUAL_M = 0.01
POSTURE_GAIN = 0.05        # per-step pull of the spare DoF towards the home posture
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
    shape = request.get("shape")
    if shape not in IMAGE_SHAPES:
        raise ValueError("shape must be [224, 224, 3] or [240, 320, 3]")
    try:
        rgb = base64.b64decode(request["image_b64"], validate=True)
    except (KeyError, TypeError, binascii.Error) as exc:
        raise ValueError("image_b64 must contain base64 RGB bytes") from exc
    if len(rgb) != math.prod(shape):
        raise ValueError(f"image_b64 must contain {shape[1]}x{shape[0]} raw RGB bytes")
    if shape != [224, 224, 3]:
        rgb = resize_image(np.frombuffer(rgb, np.uint8).reshape(shape)).tobytes()
        request["image_b64"], request["shape"] = base64.b64encode(rgb).decode(), [224, 224, 3]
    joints = finite_vector(request.get("joints_rad"), 7, "joints_rad")
    opening = request.get("gripper_opening")
    if (not isinstance(opening, (int, float)) or isinstance(opening, bool) or
            not math.isfinite(opening) or not 0 <= opening <= 1):
        raise ValueError("gripper_opening must be a number from 0 to 1")
    return joints, float(opening), rgb


def expert_action(handler, request, joints, opening, rgb):
    """Scripted expert on the privileged state, one per episode: task 1 ExpertPolicy, tasks 2-5 MultiTaskExpert."""
    task = request.get("task", 1)
    if task != 1:
        return multitask_action(handler, request, task, joints, opening, rgb)
    cube = finite_vector(request.get("cube_pos"), 3, "cube_pos")
    marker = finite_vector(request.get("marker_xy"), 2, "marker_xy")
    episode, step = request.get("episode"), request.get("step")
    if not isinstance(episode, int) or not isinstance(step, int):
        raise ValueError("expert mode needs integer episode and step")
    hand, R_hand = handler.kin.fk(joints)
    if step == 0 or handler.expert is None or handler.expert_episode != episode:
        handler.expert = ExpertPolicy(marker, R_hand)    # reset pose defines the nominal rotation
        handler.expert_episode = episode
        if handler.recorder:
            handler.recorder.begin(episode, request["instruction"], cube, marker)
    action, phase = handler.expert.act(hand, cube, opening, R_hand, step)
    if phase == "DONE":
        action = np.r_[np.zeros(6), 1.0]
    if handler.recorder:    # executed = label + DART drift; the clean label is what gets stored
        return handler.recorder.step(rgb, hand, cube, opening, action.tolist(), phase), phase
    return action.tolist(), phase


def multitask_action(handler, request, task, joints, opening, rgb):
    if task not in (2, 3, 4, 5) or not isinstance(request.get("state"), dict):
        raise ValueError("tasks 2-5 need task in 2..5 and a state object")
    episode, step = request.get("episode"), request.get("step")
    if not isinstance(episode, int) or not isinstance(step, int):
        raise ValueError("expert mode needs integer episode and step")
    state = TaskState(request["state"])
    hand, R_hand = handler.kin.fk(joints)
    if step == 0 or handler.expert is None or handler.expert_episode != episode:
        handler.expert = MultiTaskExpert(task, R_hand)
        handler.expert_episode = episode
        if handler.recorder:
            handler.recorder.begin(episode, request["instruction"], state.obj, state.target[:2], task)
    action, phase = handler.expert.act(hand, R_hand, opening, state, step)
    if handler.recorder:
        success = task_success(task, state) if phase == "DONE" else None
        return handler.recorder.step(rgb, hand, state.obj, opening, action.tolist(), phase,
                                     noise_scale=MULTI_NOISE.get(phase, 0.0), success=success), phase
    return action.tolist(), phase


def convert_action(kin, joints, opening, action):
    action = finite_vector(action, 7, "action").copy()
    distance = np.linalg.norm(action[:3])
    if distance > MAX_HAND_STEP_M:
        action[:3] *= MAX_HAND_STEP_M / distance
    action[6] = np.clip(action[6], 0.0, 1.0)
    target8, residual = decode_action(kin, np.r_[joints, opening], action, POSTURE_GAIN)
    if not math.isfinite(residual) or residual > MAX_IK_RESIDUAL_M:
        raise ValueError(f"IK residual too large: {residual:.4f} m")
    target8 = np.asarray(target8, dtype=np.float64)
    target8[:7] = np.clip(target8[:7], joints - MAX_JOINT_STEP_RAD, joints + MAX_JOINT_STEP_RAD)
    target8[:7] = np.clip(target8[:7], kin.q_lo, kin.q_hi)
    return action.tolist(), target8.tolist(), residual


def png_b64(rgb):
    """224x224 frame as lossless PNG: about a quarter of the raw bytes over the VPN."""
    buf = io.BytesIO()
    Image.fromarray(np.frombuffer(rgb, np.uint8).reshape(224, 224, 3)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class GatewayHandler(BaseHTTPRequestHandler):
    kin = None
    upstream_url = None
    llm_url = None
    save_image = None
    policy = "sample"
    expert = None
    expert_episode = None
    recorder = None
    requests = 0
    lock = threading.Lock()

    def log_message(self, *args):
        pass    # one access line per 10 Hz step is noise; progress is printed below

    def send_json(self, code, value):
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward_interpret(self):
        """Operator text -> the server's language front end (task number, trained sentence, reply)."""
        if not self.llm_url:
            return self.send_json(503, {"error": "no --llm-url"})
        try:
            body = self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 10000))
            req = urllib.request.Request(self.llm_url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                out = json.load(response)
            print(f"interpret -> task {out.get('task')}: {out.get('reply')}", flush=True)
            self.send_json(200, out)
        except (ValueError, urllib.error.URLError, OSError) as exc:
            self.send_json(502, {"error": str(exc)})

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self.send_json(200, {"ok": True, "mode": "upstream" if self.upstream_url else self.policy,
                                         "requests": self.requests})
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/interpret":
            return self.forward_interpret()
        if self.path != "/act":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            joints, opening, rgb = validate_request(request)
            if self.save_image:     # the exact 224x224 frame the policy sees
                Image.fromarray(np.frombuffer(rgb, np.uint8).reshape(224, 224, 3)).save(self.save_image)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)})

        start = time.monotonic()
        phase = ""
        try:
            if self.upstream_url:
                upstream_request = urllib.request.Request(
                    self.upstream_url,
                    json.dumps({"instruction": request["instruction"], "shape": request["shape"],
                                "image_png_b64": png_b64(rgb)}).encode(),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(upstream_request, timeout=30) as response:
                    action = json.load(response)["action"]
            with self.lock:
                if not self.upstream_url and self.policy == "expert":
                    action, phase = expert_action(type(self), request, joints, opening, rgb)
                elif not self.upstream_url:
                    action = SAMPLE_ACTION
                safe_action, targets, residual = convert_action(self.kin, joints, opening, action)
                type(self).requests += 1
            if self.requests % 25 == 1 or self.policy == "sample":
                print(f"request {self.requests}: step {request.get('step')} {phase} "
                      f"IK residual {residual:.6f} m", flush=True)
            self.send_json(200, {"action": safe_action, "joint_targets": targets,
                                 "ik_residual_m": residual, "phase": phase,
                                 "latency_s": round(time.monotonic() - start, 3)})
        except (ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
            self.send_json(502, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=("sample", "expert"), default="sample",
                        help="local action source when --upstream-url is not given")
    parser.add_argument("--upstream-url", help="OpenVLA POST /act endpoint (overrides --policy)")
    parser.add_argument("--record", help="expert mode: save DART demonstrations to this folder")
    parser.add_argument("--seed", type=int, default=1, help="recording noise seed (one manifest per seed)")
    parser.add_argument("--llm-url", help="language front end POST /interpret (default: next to --upstream-url)")
    parser.add_argument("--save-image", help="overwrite this PNG with every observation (camera check)")
    args = parser.parse_args()
    GatewayHandler.kin = PandaKinematics()
    GatewayHandler.upstream_url = args.upstream_url
    GatewayHandler.llm_url = args.llm_url or (
        args.upstream_url.rsplit("/", 1)[0] + "/interpret" if args.upstream_url else None)
    GatewayHandler.policy = args.policy
    GatewayHandler.save_image = args.save_image
    if args.record:
        if args.upstream_url or args.policy != "expert":
            parser.error("--record needs --policy expert")
        GatewayHandler.recorder = DemoRecorder(args.record, args.seed)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), GatewayHandler)
    print(f"IK_GATEWAY_READY http://127.0.0.1:{args.port} mode="
          f"{'upstream' if args.upstream_url else args.policy}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
