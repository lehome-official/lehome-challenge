"""
LeHome Challenge — Your Policy.

EDIT THIS FILE to implement your policy.

Replace the DummyPolicy class with your own model logic.
The server handles all communication — you only need to implement:
    reset()  — called at the start of each episode
    infer()  — receives observations, returns actions
"""

import numpy as np
from typing import Dict, List
from server import BasePolicyServer

ACTION_DIM = 12
CHUNK_SIZE = 10


class DummyPolicy(BasePolicyServer):
    """
    Dummy policy that returns zero actions.

    Replace this with your own model. For example:
        def __init__(self):
            self.model = load_your_model("checkpoint.pt")

        def infer(self, observation):
            images = observation["observation.images.top_rgb"]   # (H, W, 3) uint8
            state = observation["observation.state"]             # (12,) float32
            actions = self.model.predict(images, state)
            return [actions]  # single action, or list of N for chunking
    """

    def reset(self):
        print("  Episode reset")

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        for key, value in observation.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"min={value.min():.3f}, max={value.max():.3f}")

        actions = [np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(CHUNK_SIZE)]
        print(f"  -> returning {len(actions)} actions, dim={ACTION_DIM}")
        return actions


if __name__ == "__main__":
    DummyPolicy().run()
