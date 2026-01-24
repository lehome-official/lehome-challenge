"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .example_participant_policy import CustomPolicy

__all__ = [
    "BasePolicy", 
    "PolicyRegistry",
    "LeRobotPolicy", 
    "CustomPolicy", 
]
