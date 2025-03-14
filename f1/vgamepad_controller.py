"""
pyautogui_controller.py

Maps actions like FORWARD, LEFT, RIGHT, STOP to keyboard presses 
using pyautogui. Useful on macOS (or any OS) where vgamepad isn't supported.

Requirements:
    pip install pyautogui
    # On macOS, may also need to enable "Accessibility" in System Preferences.
"""

import time
import pyautogui

def handle_pyautogui_action(action: str):
    """
    Map recognized gesture ("FORWARD", "LEFT", "RIGHT", "STOP", etc.)
    to keyboard commands.

    By default:
      FORWARD => 'w'
      LEFT    => 'a'
      RIGHT   => 'd'
      STOP    => 's'
    """

    # Key mappings (you can tweak these to suit your game)
    forward_key = 'w'
    left_key    = 'a'
    right_key   = 'd'
    stop_key    = 's'

    # Press or type based on the action. 
    # For a quick press, we'll do press(...) or keyDown/keyUp logic.
    if action == "FORWARD":
        # quick press
        pyautogui.press(forward_key)
        # or hold down for a bit:
        # pyautogui.keyDown(forward_key)
        # time.sleep(0.1)
        # pyautogui.keyUp(forward_key)

    elif action == "LEFT":
        pyautogui.press(left_key)

    elif action == "RIGHT":
        pyautogui.press(right_key)

    elif action == "STOP":
        # Press 's' or do something else if you want "stop" logic
        pyautogui.press(stop_key)

    # else: no recognized action => do nothing
