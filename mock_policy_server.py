"""Dependency-free stand-in for policy_server.py's HTTP inference endpoint.

Run: python3 mock_policy_server.py --port 8000
This checks the raw RGB request and returns a stationary 7D action. When
Unreal supplies joint state, it also returns a small mock-only joint target.
It does not load OpenVLA.
"""

import argparse
import base64
import binascii
import json
import math
import struct
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MAX_BODY_BYTES = 1_000_000
IMAGE_SHAPE = [224, 224, 3]


def save_rgb_png(path, rgb):
    """Write the exact RGB bytes sent by Unreal, without image dependencies."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + rgb[y * 224 * 3:(y + 1) * 224 * 3] for y in range(224))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


class MockPolicyHandler(BaseHTTPRequestHandler):
    request_count = 0
    image_path = Path("doc/mock_policy_observation.png")

    def send_json(self, status, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self.send_json(200, {
                "ok": True,
                "adapter": "mock",
                "requests": self.request_count,
                "next": "Unreal sends a POST to /act; opening /act in a browser will not run the policy",
            })
        self.send_json(404, {"error": "not found", "available": ["GET /health", "POST /act"]})

    def do_POST(self):
        if self.path != "/act":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request.get("instruction"), str) or not request["instruction"].strip():
                raise ValueError("instruction must be nonempty text")
            if request.get("shape") != IMAGE_SHAPE:
                raise ValueError("shape must be [224, 224, 3]")
            image = base64.b64decode(request["image_b64"], validate=True)
            if len(image) != 224 * 224 * 3:
                raise ValueError("image_b64 must contain 224x224 raw RGB bytes")
            joints = request.get("joints_rad")
            if joints is not None:
                if (not isinstance(joints, list) or len(joints) != 7 or
                        any(not isinstance(q, (int, float)) or isinstance(q, bool) or
                            not math.isfinite(q) for q in joints)):
                    raise ValueError("joints_rad must contain seven finite numbers")
                opening = request.get("gripper_opening")
                if (not isinstance(opening, (int, float)) or isinstance(opening, bool) or
                        not math.isfinite(opening) or not 0 <= opening <= 1):
                    raise ValueError("gripper_opening must be a number from 0 to 1")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
            return self.send_json(400, {"error": str(exc)})

        type(self).request_count += 1
        try:
            save_rgb_png(self.image_path, image)
        except OSError as exc:
            print(f"Could not save observation: {exc}", flush=True)
        print(f"request {self.request_count}: {request['instruction']} ({len(image)} RGB bytes)", flush=True)
        print(f"latest observation: {self.image_path}", flush=True)
        # Seven OpenVLA-style values remain stationary. The separate eight-value
        # target is only for testing Unreal's command path, not policy inference.
        response = {"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "latency_s": 0.0}
        if joints is not None:
            targets = list(joints)
            targets[0] = min(targets[0] + 0.1, 2.8973)
            response["joint_targets"] = targets + [opening]
        self.send_json(200, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save-image", type=Path, default=MockPolicyHandler.image_path)
    args = parser.parse_args()
    MockPolicyHandler.image_path = args.save_image
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MockPolicyHandler)
    print(f"MOCK_POLICY_READY http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
