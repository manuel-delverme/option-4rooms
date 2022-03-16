import numpy as np

import emdp.actions
import emdp.gridworld
import hyper

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


def make_env(task_idx) -> emdp.gridworld.GridWorldMDP:
    env = make_4rooms(task_idx)
    return env


def make_2states():
    ascii_room = """
    ####
    #  #
    # ##
    ####
    """.split('\n')[1:-1]

    goal = (1, 3)
    env = NumpyWrapper(goal=goal, ascii_room=ascii_room)
    return env


class NumpyWrapper(emdp.gridworld.GridWorldMDP):
    def step(self, action):
        return super(NumpyWrapper, self).step(int(action))


def make_4rooms(task_idx):
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

    goal_list = [(1, 7), (7, 1), (7, 7), (1, 1), ][:hyper.num_tasks]
    env = emdp.gridworld.GridWorldMDP(goals=goal_list, ascii_room=ascii_room, rgb_features=False, forced_goal=task_idx)
    for idx, g1 in enumerate(goal_list):
        for g2 in goal_list:
            if g1 == g2:
                continue
            g2s = env.flatten_state(g2).argmax()
            env.rewards[idx][g2s, env.rewarding_action] = -1
            env.terminal_matrices[idx][g2s, env.rewarding_action] = True
    return env
