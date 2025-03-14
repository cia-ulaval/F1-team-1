"""
vgamepad_controller.py

Emulates an Xbox 360 gamepad using vgamepad.
Maps actions like FORWARD, LEFT, RIGHT, STOP to joystick axis.
Uses the alias: gamepad = vg.VX360Gamepad()

Requirements:
    pip install vgamepad
    (Windows only) Install ViGEmBus driver:
      https://github.com/ViGEm/ViGEmBus/releases
"""

import time
import vgamepad as vg  

gamepad = vg.VX360Gamepad()

def handle_vgamepad_action(action: str):
    """
    Map recognized gesture (FORWARD, LEFT, RIGHT, STOP, etc.)
    to vgamepad inputs on the left joystick.

    By default:
      FORWARD => push joystick up
      LEFT => push joystick left
      RIGHT => push joystick right
      STOP => center
    """

    x_val = 0.0
    y_val = 0.0

    if action == Movement.FORWARD:
        y_val = -1.0
    elif action == Movement.LEFT:
        x_val = -1.0
    elif action == Movement.RIGHT:
        x_val = 1.0
    elif action == Movement.STOP:
        x_val = 0.0
        y_val = 0.0

    gamepad.left_joystick_float(x_value_float=x_val,
                                y_value_float=y_val)

    gamepad.update()


