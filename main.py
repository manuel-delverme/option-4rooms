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

import hyper
import metrics
import option_baselines.aoc
import option_baselines.common.buffers
import option_baselines.common.callbacks
import option_baselines.common.torch_layers
import task

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
                raise NotImplementedError
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


class PolicyHideTask(stable_baselines3.common.policies.MultiInputActorCriticPolicy):
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        obs = {
            "image": obs["image"],
            "task": obs["task"] * 0,
        }
        obs = super().extract_features(obs)
        return obs


def main():
    envs = wrap_envs(hyper.num_tasks, hyper.num_envs_per_task)
    agent = option_baselines.aoc.AOC(
        meta_policy=MetaActorCriticPolicy,
        policy=PolicyHideTask,
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
        metrics.CallBack(),
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
    tb = experiment_buddy.deploy()
    main(tb, torch.device("cpu"))
