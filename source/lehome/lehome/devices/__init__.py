from .device_base import DeviceBase
import os

# Lazy imports for hardware devices that may fail in headless environments
# SO101Leader requires pynput which needs X server
if os.environ.get("LEHOME_DISABLE_SO101") != "1":
    try:
        from .lerobot import SO101Leader, BiSO101Leader
    except ImportError:
        SO101Leader = None
        BiSO101Leader = None

if os.environ.get("LEHOME_DISABLE_KEYBOARD") != "1":
    try:
        from .keyboard import Se3Keyboard, BiKeyboard
    except ImportError:
        Se3Keyboard = None
        BiKeyboard = None

__all__ = [
    "DeviceBase",
    "SO101Leader",
    "BiSO101Leader",
    "Se3Keyboard",
    "BiKeyboard",
]
