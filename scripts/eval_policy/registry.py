"""
Policy Registry for LeHome Challenge

This module provides a registry system for policies, allowing participants
to register their custom policies without modifying core evaluation code.
"""

from typing import Dict, Type, Optional
from .base_policy import BasePolicy


class PolicyRegistry:
    """
    Global registry for all available policies.
    
    Usage:
        # Register a policy
        @PolicyRegistry.register("my_policy")
        class MyPolicy(BasePolicy):
            pass
        
        # Or register manually
        PolicyRegistry.register_policy("my_policy", MyPolicy)
        
        # Get available policies
        available = PolicyRegistry.list_policies()
        
        # Create policy instance
        policy = PolicyRegistry.create("my_policy", model_path="...")
    """
    
    _registry: Dict[str, Type[BasePolicy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a policy class.
        
        Args:
            name: Unique identifier for the policy.
            
        Example:
            @PolicyRegistry.register("my_policy")
            class MyPolicy(BasePolicy):
                pass
        """
        def decorator(policy_cls: Type[BasePolicy]):
            cls.register_policy(name, policy_cls)
            return policy_cls
        return decorator
    
    @classmethod
    def register_policy(cls, name: str, policy_cls: Type[BasePolicy]):
        """
        Manually register a policy class.
        
        Args:
            name: Unique identifier for the policy.
            policy_cls: Policy class (must inherit from BasePolicy).
            
        Raises:
            ValueError: If policy name already exists or class doesn't inherit BasePolicy.
        """
        if name in cls._registry:
            raise ValueError(f"Policy '{name}' is already registered!")
        
        if not issubclass(policy_cls, BasePolicy):
            raise ValueError(f"Policy class must inherit from BasePolicy, got {policy_cls}")
        
        cls._registry[name] = policy_cls
        print(f"[PolicyRegistry] Registered policy: '{name}' -> {policy_cls.__name__}")
    
    @classmethod
    def get_policy_class(cls, name: str) -> Type[BasePolicy]:
        """
        Get policy class by name.
        
        Args:
            name: Policy identifier.
            
        Returns:
            Policy class.
            
        Raises:
            KeyError: If policy name not found.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Policy '{name}' not found in registry. "
                f"Available policies: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BasePolicy:
        """
        Create a policy instance by name.
        
        Args:
            name: Policy identifier.
            **kwargs: Arguments to pass to policy constructor.
            
        Returns:
            Policy instance.
        """
        policy_cls = cls.get_policy_class(name)
        return policy_cls(**kwargs)
    
    @classmethod
    def list_policies(cls) -> list:
        """
        Get list of all registered policy names.
        
        Returns:
            List of policy names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a policy is registered.
        
        Args:
            name: Policy identifier.
            
        Returns:
            True if registered, False otherwise.
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls):
        """Clear all registered policies (mainly for testing)."""
        cls._registry.clear()
