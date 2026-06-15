"""
Canonical action definition for the Super Mario project.

Every student model and every evaluator script must use this order.  The order
matches gym_super_mario_bros.actions.COMPLEX_MOVEMENT so that the agent can
clear ANY of the 32 stages (1-1 ~ 8-4), not just simple right-running levels.

"""

from __future__ import annotations

ACTION_MEANINGS = (
    "NOOP",  # 0
    "RIGHT",  # 1
    "RIGHT_JUMP",  # 2  right + A
    "RIGHT_RUN",  # 3  right + B
    "RIGHT_RUN_JUMP",  # 4  right + A + B
    "JUMP",  # 5  A
    "LEFT",  # 6
    "LEFT_JUMP",  # 7  left + A
    "LEFT_RUN",  # 8  left + B
    "LEFT_RUN_JUMP",  # 9  left + A + B
    "DOWN",  # 10
    "UP",  # 11
)

# Button combinations passed to nes_py JoypadSpace.
# Identical (order + contents) to gym_super_mario_bros COMPLEX_MOVEMENT.
MARIO_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["up"],
]


def describe_actions() -> dict[int, str]:
    return {i: name for i, name in enumerate(ACTION_MEANINGS)}
