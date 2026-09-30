#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time
from queue import Queue
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardHILTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
mouse = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard, mouse
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardHILTeleop(KeyboardEndEffectorTeleop):
    """Keyboard intervention device for HIL-SERL, in place of a leader arm.

    Unlike `KeyboardEndEffectorTeleop` (intervening only while a movement key is held), Space toggles
    the intervention like with the leader arm, so the policy cannot grab control back between two
    key presses: while intervening and no key is pressed, the arm holds still.

    The mouse (and the arrow keys) move in the plane of the table as seen by the operator:
    screen-up moves along `view_azimuth_deg`, screen-right 90° clockwise of it. Mouse motion is
    proportional: `mouse_pixels_per_step` pixels within one control step give a full step.
    Left / right Shift move down / up, left / right Ctrl close / open the gripper.
    s = success, Esc = failure, r = re-record.
    """

    config_class = KeyboardHILTeleopConfig
    name = "keyboard_hil"

    def __init__(self, config: KeyboardHILTeleopConfig):
        super().__init__(config)
        self.is_intervention = False
        self.control_events = Queue()
        self.mouse_listener = None
        self._mouse_lock = threading.Lock()
        self._mouse_last: tuple[float, float] | None = None
        self._mouse_motion = [0.0, 0.0]

    def connect(self) -> None:
        super().connect()
        if self.config.use_mouse_xy and self.listener is not None and mouse is not None:
            self.mouse_listener = mouse.Listener(on_move=self._on_move)
            self.mouse_listener.start()
            logging.info("Mouse x-y control enabled while intervening.")

    def disconnect(self) -> None:
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
            self.mouse_listener = None
        super().disconnect()

    def _on_move(self, x, y):
        with self._mouse_lock:
            if self._mouse_last is not None:
                self._mouse_motion[0] += x - self._mouse_last[0]
                self._mouse_motion[1] += y - self._mouse_last[1]
            self._mouse_last = (x, y)

    def _take_mouse_motion(self) -> tuple[float, float]:
        """Screen-pixel motion since the last call (right, down)."""
        with self._mouse_lock:
            dx, dy = self._mouse_motion
            self._mouse_motion = [0.0, 0.0]
        return dx, dy

    def _on_press(self, key):
        if key == keyboard.Key.space:
            self.is_intervention = not self.is_intervention
            # Motion made before taking over must not turn into a jump
            self._take_mouse_motion()
            logging.info(f"Intervention {'ON' if self.is_intervention else 'OFF'}")
            return
        if key == keyboard.Key.esc:
            self.control_events.put(TeleopEvents.TERMINATE_EPISODE)
            return
        char = getattr(key, "char", None)
        if char == "s":
            self.control_events.put(TeleopEvents.SUCCESS)
        elif char == "r":
            self.control_events.put(TeleopEvents.RERECORD_EPISODE)
        else:
            self.event_queue.put((key, True))

    def _on_release(self, key):
        # Esc ends the episode here; it must not disconnect the keyboard like in the parent class
        self.event_queue.put((key, False))

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        pressed = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        def axis(negative, positive) -> float:
            return float((positive in pressed) - (negative in pressed))

        screen_up = axis(keyboard.Key.down, keyboard.Key.up)
        screen_right = axis(keyboard.Key.left, keyboard.Key.right)
        mouse_right, mouse_down = self._take_mouse_motion()
        if self.is_intervention:
            screen_up -= mouse_down / self.config.mouse_pixels_per_step
            screen_right += mouse_right / self.config.mouse_pixels_per_step

        # Screen axes -> robot base frame: up = (cos a, sin a), right = (sin a, -cos a)
        azimuth = np.deg2rad(self.config.view_azimuth_deg)
        delta_x = screen_up * np.cos(azimuth) + screen_right * np.sin(azimuth)
        delta_y = screen_up * np.sin(azimuth) - screen_right * np.cos(azimuth)

        action = {
            "delta_x": float(np.clip(delta_x, -1.0, 1.0)),
            "delta_y": float(np.clip(delta_y, -1.0, 1.0)),
            "delta_z": axis(keyboard.Key.shift, keyboard.Key.shift_r),
        }
        if self.config.use_gripper:
            # Discrete command as consumed by GripperVelocityToJoint on SO101 (and recorded by the
            # leader arm): 0 raises the position (opens), 1 stays, 2 lowers it (closes)
            action["gripper"] = 1.0 + axis(keyboard.Key.ctrl_r, keyboard.Key.ctrl_l)
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        events = set()
        while not self.control_events.empty():
            events.add(self.control_events.get_nowait())
        rerecord = TeleopEvents.RERECORD_EPISODE in events
        return {
            TeleopEvents.IS_INTERVENTION: self.is_intervention,
            TeleopEvents.TERMINATE_EPISODE: rerecord or TeleopEvents.TERMINATE_EPISODE in events,
            TeleopEvents.SUCCESS: TeleopEvents.SUCCESS in events,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }


class KeyboardTeleopEvents:
    """Keyboard episode controls for teleoperators without buttons (e.g. a leader arm in HIL-SERL).

    Key bindings follow the HIL-SERL docs:
    - Space: toggle intervention (take over / give control back to the policy)
    - s: mark the episode as a success
    - Esc: terminate the episode as a failure
    - r: re-record the episode
    """

    def __init__(self):
        self.listener = None
        self.event_queue = Queue()
        self.is_intervention = False

    def start(self) -> None:
        if self.listener is not None:
            return
        if not (PYNPUT_AVAILABLE and pynput_can_capture()):
            logging.warning(
                "Keyboard events are unavailable in this environment (pynput needs an X11 session on "
                "Linux). Intervention, success and failure keys will not work."
            )
            return
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        logging.info("Keyboard events enabled: Space=intervention, s=success, Esc=failure, r=re-record")

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _on_press(self, key) -> None:
        if key == keyboard.Key.space:
            self.is_intervention = not self.is_intervention
            logging.info(f"Intervention {'ON' if self.is_intervention else 'OFF'}")
        elif key == keyboard.Key.esc:
            self.event_queue.put(TeleopEvents.TERMINATE_EPISODE)
        elif getattr(key, "char", None) == "s":
            self.event_queue.put(TeleopEvents.SUCCESS)
        elif getattr(key, "char", None) == "r":
            self.event_queue.put(TeleopEvents.RERECORD_EPISODE)

    def get_events(self) -> dict[str, Any]:
        self.start()
        pressed = set()
        while not self.event_queue.empty():
            pressed.add(self.event_queue.get_nowait())

        success = TeleopEvents.SUCCESS in pressed
        rerecord = TeleopEvents.RERECORD_EPISODE in pressed
        return {
            TeleopEvents.IS_INTERVENTION: self.is_intervention,
            TeleopEvents.TERMINATE_EPISODE: rerecord or TeleopEvents.TERMINATE_EPISODE in pressed,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }
