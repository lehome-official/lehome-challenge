# Policy Integration Guide

Quick guide for evaluating policies in the LeHome Challenge.

---

## Using LeRobot Policies

Evaluate trained LeRobot models (ACT, Diffusion Policy, SmolVLA):

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_fold/checkpoints/100000/pretrained_model \
    --dataset_root Datasets/record/001 \
    --stage release \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

> 💡 **Tip:** Using `--device cpu` ensures the simulator runs on CPU to avoid GUI conflicts in some environments, while the actual policy (LeRobot or Custom) will still be loaded to CUDA for fast inference. `--enable_cameras` is required to see camera views in the GUI or record videos.

**Requirements:**
- `pretrained_model` directory with config files
- Training dataset metadata (for `--dataset_root`)
- For VLA models: add `--task_description "fold the garment on the table"`

---

## Creating Custom Policies

### Three Simple Steps

**1. Create Policy Class**

```python
# my_policy.py
import numpy as np
from typing import Dict
from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.registry import PolicyRegistry

@PolicyRegistry.register("my_policy")
class MyPolicy(BasePolicy):
    def __init__(self, model_path=None, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        # Load your model here
        
    def reset(self):
        """Called at the start of each episode."""
        pass
    
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            observation: Dict with keys like:
                - "observation.state": (N,) float32 - joint angles
                - "observation.images.top": (H,W,3) uint8 - top camera
                - "observation.images.wrist_left/right": (H,W,3) uint8
        
        Returns:
            action: (action_dim,) float32 - joint angle commands
                Single-arm: (6,), Dual-arm: (12,)
                Dim order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        # Your inference logic
        action = self.model.predict(observation)
        return action.astype(np.float32)
```

**2. Register Policy**

Just add `@PolicyRegistry.register("my_policy")` decorator (already shown above).

**3. Import in __init__.py**

```python
# scripts/eval_policy/__init__.py
from .my_policy import MyPolicy
```

### Evaluate Your Policy

```bash
python -m scripts.eval \
    --policy_type my_policy \
    --stage release \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

---

## Policy Requirements

Your custom policy must:

1. ✅ Inherit from `BasePolicy`
2. ✅ Implement `select_action(observation: Dict) -> np.ndarray`
3. ✅ Return actions as `float32` numpy array
4. ✅ Handle action dimensions: (6) single-arm or (12) dual-arm

Optional:
- Implement `reset()` to clear temporal buffers

---

## Quick Examples

**Test with built-in policies:**

```bash
# Random policy (evaluates default garments in the release stage)
python -m scripts.eval --policy_type custom --stage release --garment_type tops_long --num_episodes 1 --enable_cameras --device cpu
```

**Record evaluation videos:**

```bash
python -m scripts.eval \
    --policy_type my_policy \
    --policy_path models/my_model.pth \
    --stage release \
    --save_video \
    --video_dir outputs/eval_videos \
    --enable_cameras \
    --device cpu
```

---

## Tips

**Processing images:**
```python
import torch
image = observation["observation.images.top"]  # (H,W,3) uint8
image_tensor = torch.from_numpy(image).permute(2,0,1) / 255.0  # (C,H,W) [0,1]
```

**Handling temporal information:**
```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.obs_buffer = []

def reset(self):
    self.obs_buffer.clear()

def select_action(self, observation):
    self.obs_buffer.append(observation)
    # Use buffer for RNN/LSTM inference
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Policy not found | Check decorator `@PolicyRegistry.register("name")` and import in `__init__.py` |
| Action dimension error | Verify action shape: (6) for single-arm, (12) for dual-arm |
| LeRobot fails to load | Ensure `--policy_path` points to `pretrained_model` dir and `--dataset_root` is correct |

---

## Reference Files

- [example_participant_policy.py](example_participant_policy.py) - Complete implementation examples
- [base_policy.py](base_policy.py) - Interface definition
- [registry.py](registry.py) - Registration system

---

For more help, visit [LeHome Challenge](https://lehome-challenge.com/).
