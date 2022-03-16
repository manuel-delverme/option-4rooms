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


class MetaActorCriticPolicy(stable_baselines3.common.policies.ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # kwargs["features_extractor_class"] = MetaCombinedExtractor
        kwargs["net_arch"] = []
        super(MetaActorCriticPolicy, self).__init__(*args, **kwargs)


def main(buddy_writer):
    envs = stable_baselines3.common.vec_env.DummyVecEnv(env_fns=[task.make_2states], )
    envs.seed(hyper.seed)
    agent = option_baselines.aoc.AOC(
        meta_policy=MetaActorCriticPolicy,
        policy=stable_baselines3.common.policies.ActorCriticPolicy,
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


if __name__ == "__main__":
    np.random.seed(hyper.seed)
    torch.manual_seed(hyper.seed)
    experiment_buddy.register_defaults(vars(hyper))
    tb = experiment_buddy.deploy(wandb_kwargs={"mode": "disabled"})
    main(tb)
