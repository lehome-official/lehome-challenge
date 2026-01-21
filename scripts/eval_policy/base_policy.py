import abc
import numpy as np
from typing import Dict, Any

class BasePolicy(abc.ABC):
    """
    Base Policy Class for LeHome Challenge.
    
    All participant submissions must inherit from this class and implement 
    the `select_action` and `reset` methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the policy. Model weights and configurations should be 
        loaded here.
        """
        pass

    def reset(self):
        """
        Reset the policy state.
        
        Called at the beginning of each episode (e.g., to clear RNN 
        hidden states or action buffers).
        """
        pass

    @abc.abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate an action based on the given observation.

        Args:
            observation (Dict[str, np.ndarray]): Environmental observation data (Numpy format).
                - Images: (H, W, C), uint8, range [0, 255]
                - States: (N,), float32
                
        Returns:
            action (np.ndarray): Action command (Numpy format, float32).
        """
        raise NotImplementedError("The select_action method must be implemented.")