import torch
import numpy as np
from typing import Dict, Any, Optional, Set, Union
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor.core import TransitionKey

from lehome.utils.logger import get_logger
from scripts.utils.eval_utils import preprocess_observation
from .base_policy import BasePolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


@PolicyRegistry.register("lerobot")
class LeRobotPolicy(BasePolicy):
    """
    Adapter class for official LeRobot policies (ACT, Diffusion, SmolVLA, etc.).
    
    This class handles:
    1. Loading policy weights and configurations.
    2. Filtering observation keys to match policy requirements.
    3. Preprocessing raw numpy observations into tensors (including image normalization).
    4. Running inference.
    5. Postprocessing actions (un-normalization).
    """

    def __init__(
        self, 
        policy_path: str, 
        dataset_root: str, 
        task_description: str, 
        device: str = "cuda"
    ):
        """
        Initialize the LeRobot policy.

        Args:
            policy_path: Path to the pretrained model checkpoint.
            dataset_root: Path to the dataset root (used for metadata).
            task_description: Text description of the task (for VLA models).
            device: Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = torch.device(device)
        self.task_description = task_description
        
        logger.info(f"Loading LeRobot policy from: {policy_path}")
        
        # 1. Load Metadata
        meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
        
        # 2. Load Policy Config
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides={})
        policy_cfg.pretrained_path = policy_path
        
        # 3. Filter Metadata (Logic from original create_il_policy)
        # Identify features required by the policy
        self.input_features: Optional[Set[str]] = None
        if hasattr(policy_cfg, "input_features"):
            self.input_features = set(policy_cfg.input_features.keys())
            self._filter_metadata(meta, self.input_features)

        # 4. Create Policy
        self.policy = make_policy(policy_cfg, ds_meta=meta)
        self.policy.eval()
        self.policy.to(self.device)
        
        # 5. Create Processors
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        
        # 6. Infer Action Dimension (Logic from original run_evaluation_loop)
        self.action_dim = self._infer_action_dim(meta, task_description)
        logger.info(f"LeRobotPolicy initialized. Action dim: {self.action_dim}")

    def reset(self):
        """Reset the internal state of the policy."""
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate action from observation.

        Args:
            observation: Dictionary of numpy arrays (raw environment output).

        Returns:
            action: Numpy array of action values (un-normalized).
        """
        # 1. Filter observations (keep only what the policy needs)
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)

        # 2. Preprocess (Numpy -> Tensor Batch, Normalize, etc.)
        batch_obs = self._process_observation(observation)
        
        # 3. Inference
        with torch.inference_mode():
            batch_action = self.policy.select_action(batch_obs)
            
        # 4. Postprocess (Un-normalize)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)
            
        # 5. Convert to Numpy (Remove batch dimension)
        return batch_action.squeeze(0).cpu().numpy()

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    def _filter_metadata(self, meta: LeRobotDatasetMetadata, expected_keys: Set[str]):
        """Remove extra features from metadata that are not required by the policy."""
        dataset_features = set(meta.features.keys())
        
        # Find features in dataset but not needed by policy
        extra_features = dataset_features - expected_keys
        
        # Filter out system features (these are OK to have extra)
        system_features = {
            "timestamp", "frame_index", "episode_index", 
            "index", "task_index", "next.done"
        }
        extra_features = extra_features - system_features

        # Remove extra observation features from metadata to prevent validation errors
        for feature in extra_features:
            if feature.startswith("observation."):
                del meta.features[feature]

    def _infer_action_dim(self, meta: LeRobotDatasetMetadata, task_description: str) -> int:
        """Infer action dimension from metadata or task description."""
        action_dim = None
        
        # Try metadata 'action' shape
        if meta and hasattr(meta, "features") and "action" in meta.features:
            action_shape = meta.features["action"].get("shape", [])
            if action_shape and len(action_shape) > 0:
                action_dim = action_shape[0]
                
        # Try metadata 'observation.state' shape (fallback)
        if (action_dim is None and meta and hasattr(meta, "features") 
            and "observation.state" in meta.features):
            state_shape = meta.features["observation.state"].get("shape", [])
            if action_shape and len(state_shape) > 0:
                action_dim = state_shape[0]
                
        # Final fallback based on task name (heuristic)
        if action_dim is None:
            if "Bi" in task_description or "bi" in task_description.lower():
                action_dim = 12  # Dual-arm
            else:
                action_dim = 6   # Single-arm
        
        return action_dim

    def _filter_observations(self, obs_dict: Dict[str, Any], policy_input_features: Set[str]) -> Dict[str, Any]:
        """Filter observation dictionary to only include features expected by policy."""
        filtered = {}
        for key, value in obs_dict.items():
            # Keep all non-observation keys (like internal env state if any)
            if not key.startswith("observation."):
                filtered[key] = value
            # Keep observation features that policy expects
            elif key in policy_input_features:
                filtered[key] = value
        return filtered

    def _prepare_for_preprocessor(self, observation_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation dictionary for LeRobot preprocessor pipeline."""
        obs_for_preproc = {}
        for key, value in observation_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    # (H, W, C) -> (C, H, W), [0, 1] normalization
                    value_tensor = value_tensor.permute(2, 0, 1) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
            else:
                obs_for_preproc[key] = value

        # Create transition format with complementary_data for VLA models
        dummy_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }
        return transition

    def _process_observation(self, observation_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Process observation using the LeRobot preprocessor or manual fallback."""
        if self.preprocessor is not None:
            transition = self._prepare_for_preprocessor(observation_dict)
            transformed_transition = self.preprocessor._forward(transition)
            return self.preprocessor.to_output(transformed_transition)
        else:
            # Fallback to manual preprocessing (moved to utils in Step A)
            return preprocess_observation(
                observation_dict, self.device, self.task_description
            )