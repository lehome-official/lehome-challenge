"""
Docker-based Policy for LeHome Challenge.

This policy connects to an external Docker container running a policy server
over HTTP. The simulation sends observations and receives action chunks,
cleanly separating simulation logic from policy logic.

Protocol:
    POST /reset  — called at episode start, body: {}
    POST /infer  — send observation, receive action chunk

See dummy_docker_policy for a minimal server implementation.
"""

import base64
import json
import numpy as np
import urllib.request
import urllib.error
from typing import Dict, List

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("docker")
class DockerPolicy(BasePolicy):
    """
    Policy that delegates inference to an external Docker container via HTTP.

    The container must expose two endpoints:
        POST /reset  -> {"status": "ok"}
        POST /infer  -> {"actions": [[12 floats], ...]}

    Action chunking is handled transparently: if the server returns N actions,
    the next N calls to select_action() use cached actions before re-querying.

    Usage:
        # Start your policy container first:
        #   docker run --rm -p 8080:8080 my-policy-image
        #
        # Then run evaluation:
        #   python -m scripts.eval --policy_type docker --docker_url http://localhost:8080
    """

    def __init__(self, docker_url: str = "http://localhost:8080", **kwargs):
        super().__init__(**kwargs)
        self.docker_url = docker_url.rstrip("/")
        self._action_chunk: List[np.ndarray] = []
        self._chunk_idx: int = 0

        # Verify server is reachable
        try:
            self._post("/reset", {})
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to policy server at {self.docker_url}. "
                f"Make sure your Docker container is running. Error: {e}"
            )
        print(f"[DockerPolicy] Connected to {self.docker_url}")

    def reset(self):
        """Reset policy state for a new episode."""
        self._action_chunk = []
        self._chunk_idx = 0
        self._post("/reset", {})

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get next action. Queries the server when the cached chunk is exhausted.
        """
        if self._chunk_idx >= len(self._action_chunk):
            # Need a new chunk from the server
            payload = self._serialize_observation(observation)
            response = self._post("/infer", payload)
            actions = response["actions"]
            self._action_chunk = [
                np.array(a, dtype=np.float32) for a in actions
            ]
            self._chunk_idx = 0

        action = self._action_chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action

    def _serialize_observation(self, observation: Dict[str, np.ndarray]) -> dict:
        """Serialize observation dict to JSON-compatible format."""
        payload = {}
        for key, value in observation.items():
            if not isinstance(value, np.ndarray):
                continue
            if "images" in key or "depth" in key:
                # Images and depth: encode as base64 (much smaller than JSON lists)
                payload[key] = {
                    "base64": base64.b64encode(value.tobytes()).decode("ascii"),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            else:
                payload[key] = value.tolist()
        return payload

    def _post(self, endpoint: str, data: dict) -> dict:
        """Send a POST request and return parsed JSON response."""
        url = self.docker_url + endpoint
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Policy server request failed ({url}): {e}")
