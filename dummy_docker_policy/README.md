# Docker Policy Server

This policy type completely isolates your policy from the simulator — they run as two separate processes communicating over HTTP.

Your policy runs inside a Docker container and only receives observations (images + joint states). It has no access to the simulation internals, reward functions, or success criteria. This makes submissions reproducible and easy to validate: organizers just run your container and evaluate it with the official code.

You don't need the lehome-challenge repo inside your container at all — just your model and a lightweight HTTP server.

## Quick Start

**Edit `policy.py`** — that's the only file you need to change. Replace `DummyPolicy` with your own model:

```python
class MyPolicy(BasePolicyServer):
    def __init__(self):
        self.model = load_your_model("checkpoint.pt")

    def reset(self):
        pass  # clear buffers between episodes

    def infer(self, observation):
        state = observation["observation.state"]             # (12,) float32, joint angles
        top   = observation["observation.images.top_rgb"]    # (H, W, 3) uint8
        left  = observation["observation.images.left_rgb"]   # (H, W, 3) uint8
        right = observation["observation.images.right_rgb"]  # (H, W, 3) uint8
        depth = observation["observation.top_depth"]         # (H, W) uint16, depth in mm
        prev  = observation["action"]                        # (12,) float32, previous action

        action = self.model.predict(state, top, left, right)
        return [action]  # single action, or list of N for action chunking
```

Then build and run:

```bash
docker build -t my-policy .
docker run --rm -p 8080:8080 my-policy
```

Evaluate (in another terminal):

```bash
python -m scripts.eval --policy_type docker --garment_type top_long --headless --device cpu --enable_cameras
```

## Files

| File | Edit? | Description |
|------|-------|-------------|
| `policy.py` | **Yes** | Your policy — edit this |
| `server.py` | No | HTTP server plumbing (handles communication with simulator) |
| `Dockerfile` | Maybe | Add your dependencies (`RUN pip install torch ...`) |

## Protocol Details

The server (`server.py`) exposes two HTTP POST endpoints. You don't need to change anything in this file to use `policy.py`, but it's here for reference.

### POST /reset
Called at the start of each episode.
- Request: `{}`
- Response: `{"status": "ok"}`

### POST /infer
Called when the policy needs new actions.
- Request: observation dict with keys:
  - `observation.state` — 12 floats (joint angles)
  - `observation.images.top_rgb` — `{"base64": "...", "shape": [H, W, 3], "dtype": "uint8"}`
  - `observation.images.left_rgb` — same format
  - `observation.images.right_rgb` — same format
  - `observation.top_depth` — `{"base64": "...", "shape": [H, W], "dtype": "uint16"}` (depth in mm)
  - `action` — 12 floats (previous action)
- Response: `{"actions": [[12 floats], ...]}`

Return 1 action for per-step inference, or N actions for action chunking (server is called every N steps).
