import distutils.spawn
import functools
import os

import experiment_buddy
import gym
import numpy as np
import stable_baselines3
import stable_baselines3.common.callbacks
import stable_baselines3.common.policies
import stable_baselines3.common.preprocessing
import stable_baselines3.common.torch_layers
import stable_baselines3.common.type_aliases
import stable_baselines3.common.vec_env
import torch
import torch.distributions
import torch.nn
import wandb.integration.sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import emdp
import emdp.actions
import hyper
import option_baselines.aoc
import option_baselines.common.buffers
import option_baselines.common.callbacks
import option_baselines.common.torch_layers
import task


class FCExtractor(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(FCExtractor, self).__init__(observation_space, features_dim)
        n_flatten = observation_space.sample().flatten().shape[0]
        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_flatten, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)


# TODO Lel
stable_baselines3.common.torch_layers.NatureCNN = FCExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PATH"] = f"{os.environ['PATH']}{os.pathsep}{os.environ['HOME']}/ffmpeg/ffmpeg-5.0-i686-static/"

assert distutils.spawn.find_executable("ffmpeg")


class MetaCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super(MetaCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if stable_baselines3.common.preprocessing.is_image_space(subspace):
                extractors[key] = FCExtractor(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = torch.nn.Flatten()
                total_concat_size += stable_baselines3.common.preprocessing.get_flattened_obs_dim(subspace)

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: stable_baselines3.common.type_aliases.TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class MetaActorCriticPolicy(stable_baselines3.common.policies.MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = MetaCombinedExtractor
        kwargs["net_arch"] = []
        super(MetaActorCriticPolicy, self).__init__(*args, **kwargs)


class ACP(stable_baselines3.common.policies.MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64, 64])]
        kwargs["net_arch"] = [dict(pi=[], vf=[64, 64])]
        super(ACP, self).__init__(*args, **kwargs)

        action_net = self.action_net

        class RemoveTask(torch.nn.Module):
            def forward(self, x):
                assert x.shape[1] == 83
                x2 = torch.concat([x[:, :-2], torch.zeros(x.shape[0], 2)], dim=1)
                return x2

        self.action_net = torch.nn.Sequential(
            RemoveTask(),
            torch.nn.Tanh(),
            torch.nn.Linear(83, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 83),
            action_net
        )


def enjoy(env):
    state = env.reset()
    print("==========================")
    print(f"state:\n {state}")
    print("==========================")
    while True:
        env.render()
        print("==========================")
        action = input("Action: ")
        if action == "q":
            break
        elif action == "r":
            state = env.reset()
            continue
        elif action == "d":
            action = emdp.actions.RIGHT
        elif action == "a":
            action = emdp.actions.LEFT
        elif action == "s":
            action = emdp.actions.DOWN
        elif action == "w":
            action = emdp.actions.UP
        else:
            print("Invalid action", action)
            continue

        state, reward, done, info = env.step(action)
        print("==========================")
        # print(f"state:\n {state['image'] / 10}")
        # print(f"task: {state['task']}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")

        if done:
            env.reset()


class CallBack(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super(CallBack, self).__init__()
        self.last_log = 0
        task_idx_mask = torch.div(torch.arange(hyper.num_envs), hyper.num_envs_per_task, rounding_mode="trunc")
        self.task_idx_mask = task_idx_mask.unsqueeze(0)

    def _on_step(self):
        if not (self.n_calls % 10) != 0:
            return

        meta_policy = self.locals["self"].policy.meta_policy
        obs_tensor = self.locals["obs_tensor"]
        features = meta_policy.extract_features(obs_tensor)
        latent_pi, latent_vf = meta_policy.mlp_extractor(features)
        task_idx_mask = self.task_idx_mask.squeeze()
        num_tasks = hyper.num_tasks
        distribution = meta_policy._get_action_dist_from_latent(latent_pi)
        for task_idx in range(num_tasks):
            task_dist = distribution.distribution.probs.detach()
            self.logger.add_histogram(f"task{task_idx}/meta_distr", task_dist[task_idx_mask == task_idx], self.num_timesteps)

        super(CallBack, self)._on_step()

    def _on_rollout_end(self):
        super(CallBack, self)._on_rollout_end()

        if (self.num_timesteps - self.last_log) <= hyper.log_iterate_every:
            return
        print("progress", 1 - self.locals["self"]._current_progress_remaining)
        self.last_log = self.num_timesteps
        rollout_steps = self.locals["self"].n_steps
        num_tasks = hyper.num_tasks
        rollout_buffer = self.locals["rollout_buffer"]
        executed_options = rollout_buffer.current_options
        switches = rollout_buffer.current_options != rollout_buffer.previous_options
        switches[rollout_buffer.episode_starts.astype(bool)] = False

        self.logger.add_scalar("rollout/mean_returns", rollout_buffer.returns.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/mean_rewards", rollout_buffer.rewards.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/change_points", switches.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/action_gap", np.inf, self.num_timesteps)

        self.logger.add_histogram("rollout/executed_options", executed_options, self.num_timesteps)

        env_returns = rollout_buffer.returns.mean(0)
        task_returns = env_returns.reshape(num_tasks, -1).mean(1)

        env_rewards = rollout_buffer.rewards.sum(0)
        task_rewards = env_rewards.reshape(num_tasks, -1).mean(1)

        task_idx_mask = self.task_idx_mask.repeat(rollout_steps, 1)
        for task_idx in range(num_tasks):
            task_executed_option = executed_options[task_idx_mask == task_idx]
            if len(task_executed_option) < 3:
                task_executed_option = np.tile(task_executed_option, 3)[:3]
            self.logger.add_histogram(f"task{task_idx}/task_executed_options", task_executed_option, self.num_timesteps)

            task_return = task_returns[task_idx]
            task_reward = task_rewards[0]
            self.logger.add_scalar(f"task{task_idx}/mean_return", task_return, self.num_timesteps)
            self.logger.add_scalar(f"task{task_idx}/mean_rewards", task_reward, self.num_timesteps)


def main(buddy_writer, device):
    envs = wrap_envs(hyper.num_tasks, hyper.num_envs_per_task)
    agent = option_baselines.aoc.AOC(
        meta_policy=MetaActorCriticPolicy,
        policy=ACP,
        env=envs,
        num_options=hyper.num_options,
        ent_coef=hyper.entropy_regularization,
        term_coef=hyper.termination_regularization,
        switching_margin=hyper.switching_margin,
        gamma=hyper.discount,
    )
    agent.set_logger(buddy_writer)

    cb = stable_baselines3.common.callbacks.CallbackList([
        option_baselines.common.callbacks.OptionRollout(envs, eval_freq=hyper.video_every, n_eval_episodes=hyper.num_envs),
        wandb.integration.sb3.WandbCallback(gradient_save_freq=100),
        CallBack(),
    ])
    agent.learn(hyper.training_steps, callback=cb, log_interval=(hyper.log_iterate_every // (agent.n_steps * agent.n_envs)) + 1)


last_recording = 0


def should_record_video(step):
    global last_recording
    if step - last_recording > hyper.video_every:
        last_recording = step
        return True
    return False


def wrap_envs(num_tasks, num_envs_per_task):
    indices = []
    for idx in range(num_tasks):
        indices.extend([idx] * num_envs_per_task)
    envs = stable_baselines3.common.vec_env.DummyVecEnv(env_fns=[functools.partial(task.make_env, idx) for idx in indices], )
    envs = stable_baselines3.common.vec_env.VecVideoRecorder(
        envs,
        video_folder="videos/",
        record_video_trigger=should_record_video,
        video_length=hyper.running_performance_window)
    envs.seed(hyper.seed)
    return envs


if __name__ == "__main__":
    np.random.seed(hyper.seed)
    torch.manual_seed(hyper.seed)
    experiment_buddy.register_defaults(vars(hyper))
    # proc_num = 20
    # tb = experiment_buddy.deploy("mila", "sweep.yaml", proc_num=proc_num, run_per_agent=1, disabled=hyper.DEBUG, wandb_kwargs={'project': "ogw"}, )
    # tb = experiment_buddy.deploy("mila", "sweep.yaml", proc_num=25, disabled=hyper.DEBUG)
    # tb = experiment_buddy.deploy("mila", wandb_kwargs=wandb_kwargs)
    tb = experiment_buddy.deploy()
    main(tb, torch.device("cpu"))
