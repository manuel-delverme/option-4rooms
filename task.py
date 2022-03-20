import emdp.actions
import emdp.gridworld
import gym
import gym_minigrid.envs
import gym_minigrid.minigrid
import numpy as np

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


def make_2states():
    ascii_room = """
    ####
    #  #
    # ##
    ####
    """.split('\n')[1:-1]

    goal = (1, 2)
    env = NumpyWrapper(goal=goal, ascii_room=ascii_room, rgb_features=True, initial_states=[(2, 1)])
    return env


class NumpyWrapper(emdp.gridworld.GridWorldMDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space.spaces["image"]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation["image"], reward, done, info

    def reset(self):
        observation = super().reset()
        return observation["image"]


def make_4rooms(task_idx, num_tasks):
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

    goal_list = [(1, 7), (7, 1), (7, 7), (1, 1), ][:num_tasks]
    env = emdp.gridworld.GridWorldMDP(goals=goal_list, ascii_room=ascii_room, rgb_features=False, forced_goal=task_idx)
    for idx, g1 in enumerate(goal_list):
        for g2 in goal_list:
            if g1 == g2:
                continue
            g2s = env.flatten_state(g2).argmax()
            env.rewards[idx][g2s, env.rewarding_action] = -1
            env.terminal_matrices[idx][g2s, env.rewarding_action] = True
    return env


class PakMan(gym_minigrid.envs.DynamicObstaclesEnv):
    def __init__(self, *args, **kwargs):
        super(PakMan, self).__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(self.action_space.n + 1)  # Pickup key

    def _gen_grid(self, width, height):
        super(PakMan, self)._gen_grid(width, height)
        self.place_obj(gym_minigrid.envs.Key("yellow"), max_tries=100)

    def step(self, action):
        front_cell = self.grid.get(*self.front_pos)

        # if front_cell is not None and front_cell.type != 'key' and action == self.actions.pickup:
        #     action = self.actions.forward  # Crash into it instead.
        obs, reward, done, info = super().step(action)

        touched_enemy = reward == -1 and front_cell.type == "ball"
        was_empowered = self.carrying is not None and self.carrying.type == "key"

        if self.carrying is not None and self.carrying.type != "key":
            reward = -1
            done = True
        elif was_empowered and touched_enemy:
            reward = 1
            done = False
            for o in self.obstacles:
                if (o.cur_pos == front_cell.cur_pos).all():
                    self.obstacles.remove(o)
                    break
            self.grid.set(*front_cell.cur_pos, None)
        return obs, reward, done, info


def make_pakman():
    return PakMan()


def enjoy(env):
    obs = env.reset()
    env.render()
    print(env.observation_space.spaces["image"].shape)
    for _ in range(100):
        # act_key = input("Action: ")
        # if act_key == "w":
        #     action = env.actions.forward
        # elif act_key == "a":
        #     action = env.actions.left
        # elif act_key == "d":
        #     action = env.actions.right
        # elif act_key == "e":
        #     action = env.actions.pickup
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        # print(obs["image"].shape)
        print(reward)
        print(done)
        print(info)
        print(env.carrying)
        env.render()

        if done:
            env.reset()
            env.render()


if __name__ == '__main__':
    enjoy(make_pakman())
