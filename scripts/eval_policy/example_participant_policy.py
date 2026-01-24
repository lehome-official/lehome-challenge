"""
Example: How participants can register their own policy.

This file demonstrates how a participant would create and register
their custom policy for the LeHome Challenge.
"""

import numpy as np
from typing import Dict, Optional
from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("custom")
class CustomPolicy(BasePolicy):
    """
    LeHome Challenge Custom Policy Example.
    
    This is a simple template demonstrating how to implement your own policy.
    Participants can use this class as a starting point.
    
    Usage Example:
        policy = CustomPolicy(model_path="path/to/model.pth", device="cuda")
        observation = env._get_observations()
        action = policy.select_action(observation)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", **kwargs):
        """
        Initialize the policy.
        
        Args:
            model_path: Path to model weights (optional).
                Note: Custom policies do not necessarily need to receive the model_path via command-line arguments. 
                You are free to define how your model is loaded, for example:
                - Command-line arguments
                - Reading the path from environment variables
                - Using a hard-coded path
                - Loading from a configuration file
                - Or not requiring an external model file at all
            device: Device to use ('cpu' or 'cuda').
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.device = device
        self.model_path = model_path
        
        # TODO: Load your model here
        # Note: You are not restricted to using the model_path parameter; you can define any loading logic.
        # Example 1: Load from model_path (if provided)
        # import torch
        # self.model = YourModel()
        # if model_path:
        #     checkpoint = torch.load(model_path, map_location=device)
        #     self.model.load_state_dict(checkpoint)
        # self.model.to(device)
        # self.model.eval()
        #
        # Example 2: Load from environment variables or a config file
        # import os
        # model_path = os.getenv("MY_MODEL_PATH", "default/path/to/model.pth")
        # self.model = load_model(model_path)
        #
        # Example 3: Hard-coded path
        # self.model = load_model("path/to/your/model.pth")
        
        # Example: Maintain observation history (for temporal policies like RNN/Transformer)
        self.observation_history = []
        self.max_history_length = 10
        
        print(f"[CustomPolicy] Initialized - Device: {device}, Model Path: {model_path}")
        
    def reset(self):
        """
        Reset policy state.
        Called before the start of each episode.
        """
        # Clear history buffer
        self.observation_history.clear()
        
        # TODO: Reset internal model state (if applicable)
        # Example: 
        # if hasattr(self.model, 'reset_hidden_state'):
        #     self.model.reset_hidden_state()
        
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate action based on observation.
        
        Args:
            observation: Dictionary containing environment observations:
                - "observation.state": Robot state (joint angles, etc.), shape (N,), float32
                - "observation.images.top": Top view image, shape (H, W, 3), uint8, [0-255]
                - "observation.images.wrist_left": Left wrist camera, shape (H, W, 3), uint8
                - "observation.images.wrist_right": Right wrist camera, shape (H, W, 3), uint8
                - Other sensor data...
                
        Returns:
            action: Action array, shape (action_dim,), float32
                - Single-arm task: (6,) - [6 joint angles]
                - Dual-arm task: (12,) - [6 left joints + 6 right joints]
        """
        
        # ===== Example 1: Accessing State Data =====
        if "observation.state" in observation:
            state = observation["observation.state"]  # shape: (N,)
            # TODO: Use state data
            # Example: 
            # state_tensor = torch.from_numpy(state).float().to(self.device)
            # features = self.extract_state_features(state_tensor)
            
        # ===== Example 2: Accessing Image Data =====
        images = {}
        for key in observation.keys():
            if "images" in key:
                images[key] = observation[key]  # shape: (H, W, 3), uint8
                # TODO: Preprocess image and pass to model
                # Example: 
                # import torch
                # image = observation[key]
                # # Convert (H, W, C) -> (C, H, W) and normalize to [0, 1]
                # image_tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
                # # Add batch dimension: (1, C, H, W)
                # image_tensor = image_tensor.unsqueeze(0).to(self.device) 
                # visual_features = self.model.encode_image(image_tensor)
                
        # ===== Example 3: Handling History/Time-Series =====
        # Append current observation to history
        self.observation_history.append(observation.copy())
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
            
        # TODO: Predict using history (e.g., for RNN/Transformer/LSTM)
        # Example: 
        # history_states = [obs["observation.state"] for obs in self.observation_history]
        # history_tensor = torch.from_numpy(np.stack(history_states)).float().to(self.device)
        # action_tensor = self.model.predict_from_history(history_tensor)
        # action = action_tensor.cpu().numpy()
        
        # ===== Current Implementation: Random Policy (For Demonstration) =====
        # Infer action dimension from state dimension
        if "observation.state" in observation:
            action_dim = observation["observation.state"].shape[0]
        else:
            # Default to dual-arm if state is missing
            action_dim = 12
            
        # Generate random action (small magnitude to avoid violent movement)
        # Note: Replace this with your actual model inference logic
        action = np.random.randn(action_dim).astype(np.float32) * 0.05
        
        return action
