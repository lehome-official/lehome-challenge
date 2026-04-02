import weakref
import numpy as np

from collections.abc import Callable
from pynput.keyboard import Listener, Key

import carb
import omni

from ..device_base import Device


class BiGamepad(Device):
    """Gamepad controller for bimanual SO101.

    Split bimanual mode (both arms at the same time):

        Left side (controls LEFT arm)
        - Left Stick X/Y     → shoulder_pan / shoulder_lift
        - D-Pad Up/Down      → elbow_flex
        - Left Trigger (L2)  → gripper CLOSE (analog)
        - Left Bumper  (L1)  → gripper OPEN

        Right side (controls RIGHT arm)
        - Right Stick X/Y    → shoulder_pan / shoulder_lift
        - Y / A              → elbow_flex
        - Right Trigger (R2) → gripper CLOSE (analog)
        - Right Bumper (R1)  → gripper OPEN

        Start control: START / MENU button (or keyboard B).
        Quick reset: VIEW / BACK button (or keyboard R).

    Recording keys (keyboard):
        B   → start control
        S   → start recording
        N   → mark episode success
        D   → discard episode
        ESC → abort recording

    """

    def __init__(self, env, sensitivity: float = 0.05):
        super().__init__(env)
        self.sensitivity = sensitivity

        # Disable Isaac's built-in gamepad camera control
        carb.settings.get_settings().set_bool(
            "/persistent/app/omniverse/gamepadCameraControl", False
        )

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._gamepad_sub = None
        self._keyboard_sub = None

        self._left_delta_pos = np.zeros(6)
        self._right_delta_pos = np.zeros(6)

        self.started = False
        self._reset_state = False

        # Accepts string keys ("S", "N", "D", "ESCAPE") from register_teleop_callbacks
        self._additional_callbacks: dict = {}

        self.dead_zone = 0.1
        self.axis_response_exp = 1.8
        self.axis_gain = 0.75
        self.gripper_gain = 1.1

        # Joints: 0=shoulder_pan, 1=shoulder_lift, 2=elbow_flex,
        #         3=wrist_flex, 4=wrist_roll, 5=gripper
        self._LEFT_INPUT_MAPPING = {}
        self._RIGHT_INPUT_MAPPING = {}
        # Stick directions tuned for current camera side.
        self._bind_gamepad("LEFT_STICK_RIGHT", self._LEFT_INPUT_MAPPING, 0, -1)
        self._bind_gamepad("LEFT_STICK_LEFT", self._LEFT_INPUT_MAPPING, 0, 1)
        self._bind_gamepad("LEFT_STICK_UP", self._LEFT_INPUT_MAPPING, 1, 1)
        self._bind_gamepad("LEFT_STICK_DOWN", self._LEFT_INPUT_MAPPING, 1, -1)
        # Left elbow on left-side D-pad.
        self._bind_gamepad("DPAD_UP", self._LEFT_INPUT_MAPPING, 2, 1)
        self._bind_gamepad("DPAD_DOWN", self._LEFT_INPUT_MAPPING, 2, -1)
        # Left D-pad horizontal controls wrist_flex by default.
        self._bind_gamepad("DPAD_LEFT", self._LEFT_INPUT_MAPPING, 3, -1)
        self._bind_gamepad("DPAD_RIGHT", self._LEFT_INPUT_MAPPING, 3, 1)
        self._bind_gamepad("LEFT_TRIGGER", self._LEFT_INPUT_MAPPING, 5, -1)  # close
        self._bind_gamepad_any(("LEFT_BUMPER", "LEFT_SHOULDER"), self._LEFT_INPUT_MAPPING, 5, 1)  # open

        self._bind_gamepad("RIGHT_STICK_RIGHT", self._RIGHT_INPUT_MAPPING, 0, -1)
        self._bind_gamepad("RIGHT_STICK_LEFT", self._RIGHT_INPUT_MAPPING, 0, 1)
        self._bind_gamepad("RIGHT_STICK_UP", self._RIGHT_INPUT_MAPPING, 1, 1)
        self._bind_gamepad("RIGHT_STICK_DOWN", self._RIGHT_INPUT_MAPPING, 1, -1)
        self._bind_gamepad("Y", self._RIGHT_INPUT_MAPPING, 2, 1)
        self._bind_gamepad("A", self._RIGHT_INPUT_MAPPING, 2, -1)
        # Right face buttons (Square/Circle on PS, X/B on Xbox) control wrist_flex.
        self._bind_gamepad("X", self._RIGHT_INPUT_MAPPING, 3, 1)
        self._bind_gamepad("B", self._RIGHT_INPUT_MAPPING, 3, -1)
        self._bind_gamepad("RIGHT_TRIGGER", self._RIGHT_INPUT_MAPPING, 5, -1)  # close
        self._bind_gamepad_any(("RIGHT_BUMPER", "RIGHT_SHOULDER"), self._RIGHT_INPUT_MAPPING, 5, 1)  # open

        self._current_left_values: dict = {k: 0.0 for k in self._LEFT_INPUT_MAPPING}
        self._current_right_values: dict = {k: 0.0 for k in self._RIGHT_INPUT_MAPPING}
        # Shoulder button is used for both gripper-open and wrist-rotation combo.
        self._left_shoulder_input = self._get_gamepad_input_any(
            ("LEFT_BUMPER", "LEFT_SHOULDER")
        )
        self._right_shoulder_input = self._get_gamepad_input_any(
            ("RIGHT_BUMPER", "RIGHT_SHOULDER")
        )
        self._left_wrist_combo_inputs = {
            i
            for i in (
                getattr(carb.input.GamepadInput, "DPAD_LEFT", None),
                getattr(carb.input.GamepadInput, "DPAD_RIGHT", None),
            )
            if i is not None
        }
        self._right_wrist_combo_inputs = {
            i
            for i in (
                getattr(carb.input.GamepadInput, "X", None),
                getattr(carb.input.GamepadInput, "B", None),
            )
            if i is not None
        }

        # Support different controller backends naming start/menu button.
        self._start_inputs = []
        for name in ("START", "MENU"):
            gamepad_input = getattr(carb.input.GamepadInput, name, None)
            if gamepad_input is not None:
                self._start_inputs.append(gamepad_input)
        # Use view/back for quick reset.
        self._reset_inputs = []
        for name in ("VIEW", "BACK"):
            gamepad_input = getattr(carb.input.GamepadInput, name, None)
            if gamepad_input is not None:
                self._reset_inputs.append(gamepad_input)
        self._reset_input_pressed = {inp: False for inp in self._reset_inputs}

        # Subscribe only after all state is fully initialized.
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )

        # Keyboard subscription so B/S/N/D/ESC work on XWayland too
        # (pynput global key-grabs are blocked by the Wayland compositor).
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Keyboard listener for B/S/N/D/ESC — mirrors BiKeyboard behaviour
        self._listener = Listener(on_release=self._on_key_release)
        self._listener.daemon = True
        self._listener.start()

    def __del__(self):
        if hasattr(self, "_keyboard_sub") and self._keyboard_sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if hasattr(self, "_gamepad_sub") and self._gamepad_sub is not None:
            self._input.unsubscribe_from_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None
        if hasattr(self, "_listener") and self._listener.running:
            self._listener.stop()

    def __str__(self) -> str:
        msg = "Bi-Gamepad Controller for SE(3).\n"
        try:
            msg += f"\tDevice: {self._input.get_gamepad_name(self._gamepad)}\n"
        except Exception:
            msg += "\tDevice: Unknown Gamepad\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft side → LEFT arm\n"
        msg += "\t  Left Stick X/Y    → shoulder_pan / shoulder_lift\n"
        msg += "\t  D-Pad Up/Down     → elbow_flex + / -\n"
        msg += "\t  D-Pad Left/Right  → wrist_flex + / -\n"
        msg += "\t  Hold L1 + D-Pad Left/Right → wrist_roll + / -\n"
        msg += "\t  L2 / L1           → gripper close / open\n"
        msg += "\tRight side → RIGHT arm\n"
        msg += "\t  Right Stick X/Y   → shoulder_pan / shoulder_lift\n"
        msg += "\t  Y / A             → elbow_flex + / -\n"
        msg += "\t  X / B             → wrist_flex + / - (Square / Circle on PS)\n"
        msg += "\t  Hold R1 + X/B     → wrist_roll + / - (R1 + Square/Circle on PS)\n"
        msg += "\t  R2 / R1           → gripper close / open\n"
        msg += "\tSTART/MENU          → start control\n"
        msg += "\tVIEW/BACK           → quick reset garment/env\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeyboard B   → start control\n"
        msg += "\tKeyboard R   → quick reset garment/env\n"
        msg += "\tKeyboard S   → start recording\n"
        msg += "\tKeyboard N   → mark success\n"
        msg += "\tKeyboard D   → discard episode\n"
        msg += "\tKeyboard ESC → abort\n"
        return msg

    # ------------------------------------------------------------------
    # carb.input keyboard handler (B/S/N/D/ESC — works on XWayland)
    # ------------------------------------------------------------------

    def _on_keyboard_event(self, event, *args, **kwargs):
        try:
            key_name = event.input.name if not isinstance(event.input, str) else event.input
        except AttributeError:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name == "B":
                self.started = True
                print("[BiGamepad] Control started! (keyboard B)")
            elif key_name == "R" and "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
            elif key_name == "S" and "S" in self._additional_callbacks:
                self._additional_callbacks["S"]()
            elif key_name == "N" and "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()
            elif key_name == "D" and "D" in self._additional_callbacks:
                self._additional_callbacks["D"]()
            elif key_name == "ESCAPE" and "ESCAPE" in self._additional_callbacks:
                self._additional_callbacks["ESCAPE"]()
        return True

    # ------------------------------------------------------------------
    # pynput keyboard listener (fallback for non-XWayland systems)
    # ------------------------------------------------------------------

    def _on_key_release(self, key):
        try:
            char = key.char
            if char == "b":
                self.started = True
                print("[BiGamepad] Control started! (keyboard B)")
            elif char == "r" and "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
            elif char == "s" and "S" in self._additional_callbacks:
                self._additional_callbacks["S"]()
            elif char == "n" and "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()
            elif char == "d" and "D" in self._additional_callbacks:
                self._additional_callbacks["D"]()
        except AttributeError:
            if key == Key.esc and "ESCAPE" in self._additional_callbacks:
                self._additional_callbacks["ESCAPE"]()

    # ------------------------------------------------------------------
    # Gamepad event handler
    # ------------------------------------------------------------------

    def _on_gamepad_event(self, event, *args, **kwargs):
        if not hasattr(self, "_start_inputs"):
            # Guard against callbacks fired during partial init/teardown.
            return True
        cur_val = event.value

        # Start control from dedicated controller start/menu buttons.
        if event.input in self._start_inputs:
            if cur_val > 0.5:
                self.started = True
                print("[BiGamepad] Control started!")
        # Quick environment reset from controller view/back button.
        if event.input in self._reset_inputs:
            was_pressed = self._reset_input_pressed.get(event.input, False)
            is_pressed = cur_val > 0.5
            self._reset_input_pressed[event.input] = is_pressed
            if is_pressed and not was_pressed and "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()

        # Apply dead zone to continuous inputs (sticks & triggers)
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0

        # Convenience: if control wasn't started yet, any meaningful gamepad
        # input starts teleop (avoids requiring keyboard "B").
        if (
            not self.started
            and abs(cur_val) > 0.0
            and (
                event.input in self._LEFT_INPUT_MAPPING
                or event.input in self._RIGHT_INPUT_MAPPING
            )
        ):
            self.started = True
            print("[BiGamepad] Control started! (auto from gamepad input)")

        if event.input in self._LEFT_INPUT_MAPPING:
            self._current_left_values[event.input] = cur_val
        elif event.input in self._RIGHT_INPUT_MAPPING:
            self._current_right_values[event.input] = cur_val

        return True

    def _bind_gamepad(self, name: str, mapping: dict, axis_idx: int, sign: int):
        gamepad_input = getattr(carb.input.GamepadInput, name, None)
        if gamepad_input is not None:
            mapping[gamepad_input] = (axis_idx, sign)

    def _bind_gamepad_any(self, names: tuple[str, ...], mapping: dict, axis_idx: int, sign: int):
        for name in names:
            gamepad_input = getattr(carb.input.GamepadInput, name, None)
            if gamepad_input is not None:
                mapping[gamepad_input] = (axis_idx, sign)
                return

    def _get_gamepad_input_any(self, names: tuple[str, ...]):
        for name in names:
            gamepad_input = getattr(carb.input.GamepadInput, name, None)
            if gamepad_input is not None:
                return gamepad_input
        return None

    # ------------------------------------------------------------------
    # Device interface
    # ------------------------------------------------------------------

    def get_device_state(self):
        return {
            "left_arm":  self._left_delta_pos.copy(),
            "right_arm": self._right_delta_pos.copy(),
        }

    def input2action(self):
        left_delta = np.zeros(6)
        right_delta = np.zeros(6)
        left_shoulder_pressed = (
            self._left_shoulder_input is not None
            and self._current_left_values.get(self._left_shoulder_input, 0.0) > 0.5
        )
        right_shoulder_pressed = (
            self._right_shoulder_input is not None
            and self._current_right_values.get(self._right_shoulder_input, 0.0) > 0.5
        )

        for g_input, (axis_idx, sign) in self._LEFT_INPUT_MAPPING.items():
            val = self._current_left_values[g_input]
            if val > 0:
                gain = self.gripper_gain if axis_idx == 5 else self.axis_gain
                shaped = (val ** self.axis_response_exp) * gain
                target_axis = axis_idx
                # L1 + D-pad Left/Right controls wrist_roll instead of wrist_flex.
                if (
                    left_shoulder_pressed
                    and g_input in self._left_wrist_combo_inputs
                    and axis_idx == 3
                ):
                    target_axis = 4
                # Combo takes priority: don't also issue gripper-open when using it.
                if (
                    left_shoulder_pressed
                    and self._left_shoulder_input is not None
                    and g_input == self._left_shoulder_input
                ):
                    combo_active = any(
                        self._current_left_values.get(i, 0.0) > 0.05
                        for i in self._left_wrist_combo_inputs
                    )
                    if combo_active:
                        continue
                left_delta[target_axis] += sign * shaped * self.sensitivity

        for g_input, (axis_idx, sign) in self._RIGHT_INPUT_MAPPING.items():
            val = self._current_right_values[g_input]
            if val > 0:
                gain = self.gripper_gain if axis_idx == 5 else self.axis_gain
                shaped = (val ** self.axis_response_exp) * gain
                target_axis = axis_idx
                # R1 + X/B (Square/Circle) controls wrist_roll.
                if (
                    right_shoulder_pressed
                    and g_input in self._right_wrist_combo_inputs
                    and axis_idx == 3
                ):
                    target_axis = 4
                if (
                    right_shoulder_pressed
                    and self._right_shoulder_input is not None
                    and g_input == self._right_shoulder_input
                ):
                    combo_active = any(
                        self._current_right_values.get(i, 0.0) > 0.05
                        for i in self._right_wrist_combo_inputs
                    )
                    if combo_active:
                        continue
                right_delta[target_axis] += sign * shaped * self.sensitivity

        self._left_delta_pos = left_delta
        self._right_delta_pos = right_delta

        ac_dict = {
            "reset":       self._reset_state,
            "started":     self.started,
            "bi_keyboard": True,
        }
        if self._reset_state:
            self._reset_state = False
            return ac_dict

        ac_dict["joint_state"] = self.get_device_state()
        return ac_dict

    def reset(self):
        self._left_delta_pos = np.zeros(6)
        self._right_delta_pos = np.zeros(6)
        self._current_left_values = {k: 0.0 for k in self._LEFT_INPUT_MAPPING}
        self._current_right_values = {k: 0.0 for k in self._RIGHT_INPUT_MAPPING}

    def add_callback(self, key: str, func: Callable):
        """Register a string-keyed callback (e.g. 'S', 'N', 'D', 'ESCAPE')."""
        self._additional_callbacks[key] = func
