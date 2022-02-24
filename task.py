import random

import emdp.actions
import emdp.gridworld
import gym.spaces
import numpy as np

ascii_room = """
#########
#   #   #
#       #
#   #   #
## ### ##
#   #   #
#       #
#   #   #
#########"""[1:].split('\n')

direction_ = np.zeros((4, 2, 2))
direction_[emdp.actions.LEFT] = [
    [-1, 1],
    [-1, -1]
]
direction_[emdp.actions.RIGHT] = [
    [1, -1],
    [1, 1],
]
direction_[emdp.actions.UP] = [
    [-1, -1],
    [1, -1],
]
direction_[emdp.actions.DOWN] = [
    [1, 1],
    [-1, 1]
]


def make_env(num_envs, env_idx) -> emdp.gridworld.GridWorldMDP:
    goal_list = [(1, 7), (7, 1), (7, 7), (1, 1), ]
    if num_envs > 4:
        _, goal_list = emdp.gridworld.txt_utilities.ascii_to_walls(ascii_room)
        state = random.getstate()
        random.seed(0)
        random.shuffle(goal_list)
        random.setstate(state)

    task_idx = env_idx % len(goal_list)
    goal = goal_list[task_idx]
    env = emdp.gridworld.GridWorldMDP(goal=goal, ascii_room=ascii_room)
    return env
