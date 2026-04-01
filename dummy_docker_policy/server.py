"""
LeHome Challenge — Policy Server.

DO NOT MODIFY THIS FILE. Edit policy.py instead.

Handles HTTP communication between the simulator and your policy.
See policy.py for the interface you need to implement.
"""

import base64
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List


class BasePolicyServer:
    """
    Base class for LeHome policy servers.

    Handles all HTTP plumbing. Subclass and override:
        reset()              — called at the start of each episode
        infer(observation)   — called to get the next action chunk

    Observation dict contains:
        "observation.state"           — np.ndarray, shape (12,), float32 (joint angles)
        "observation.images.top_rgb"  — np.ndarray, shape (H, W, 3), uint8
        "observation.images.left_rgb" — np.ndarray, shape (H, W, 3), uint8
        "observation.images.right_rgb"— np.ndarray, shape (H, W, 3), uint8
        "observation.top_depth"       — np.ndarray, shape (H, W), uint16 (depth in mm)
        "action"                      — np.ndarray, shape (12,), float32 (previous action)
    """

    def reset(self) -> None:
        """Called at the start of each episode. Override to clear buffers, etc."""
        pass

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Return a list of actions (action chunk).

        Args:
            observation: Dict of numpy arrays (state, images, previous action).

        Returns:
            List of np.ndarray actions, each shape (action_dim,).
            Return 1 action for per-step control, or N for action chunking.
        """
        raise NotImplementedError("Subclass must implement infer()")

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the HTTP server."""
        policy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                request = json.loads(body)

                if self.path == "/reset":
                    policy.reset()
                    response = {"status": "ok"}

                elif self.path == "/infer":
                    observation = _deserialize_observation(request)
                    actions = policy.infer(observation)
                    response = {"actions": [a.tolist() for a in actions]}

                else:
                    self.send_error(404, f"Unknown endpoint: {self.path}")
                    return

                body_bytes = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

            def log_message(self, format, *args):
                print(f"[{self.command}] {self.path}")

        server = HTTPServer((host, port), Handler)
        print(f"Policy server listening on {host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            server.server_close()


def _deserialize_observation(raw: dict) -> Dict[str, np.ndarray]:
    """Convert JSON observation to numpy arrays."""
    observation = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "base64" in value:
            buf = base64.b64decode(value["base64"])
            observation[key] = np.frombuffer(buf, dtype=value["dtype"]).reshape(value["shape"])
        elif isinstance(value, list):
            observation[key] = np.array(value, dtype=np.float32)
    return observation
