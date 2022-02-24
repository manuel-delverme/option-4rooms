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
import option_baselines.aoc
import option_baselines.common.buffers
import option_baselines.common.callbacks
import option_baselines.common.torch_layers
import task

# TODO Lel
stable_baselines3.common.torch_layers.NatureCNN = option_baselines.common.torch_layers.NatureCNN

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
                class Dummy(BaseFeaturesExtractor):
                    def forward(self, observations: torch.Tensor) -> torch.Tensor:
                        del observations
                        return torch.tensor([])

                extractors[key] = Dummy(subspace, features_dim=float("inf"))
                total_concat_size += 0
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
        kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64, 64])]
        # kwargs["net_arch"] = [dict(pi=[64, ], vf=[64, ])]
        super(ACP, self).__init__(*args, **kwargs)


def enjoy(env):
    state = env.reset()
    print("==========================")
    print(f"state:\n {state['image'] / 10}")
    print(f"task: {state['task']}")
    print("==========================")
    while True:
        env.render()
        action = input("Action: ")
        if action == "q":
            break
        elif action == "r":
            state = env.reset()
            print("==========================")
            print(f"state:\n {state['image'] / 10}")
            print(f"task: {state['task']}")
            print("==========================")
            continue
        elif action == "w":
            action = env.actions.forward
        elif action == "a":
            action = env.actions.left
        elif action == "d":
            action = env.actions.right
        else:
            action = int(action)

        state, reward, done, info = env.step(action)
        print("==========================")
        print(f"state:\n {state['image'] / 10}")
        print(f"task: {state['task']}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        print("==========================")

        if done:
            env.reset()


def main(buddy_writer, device):
    test_env = task.make_env(2, 0)
    enjoy(test_env)

    envs = wrap_envs()

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

    class CallBack(stable_baselines3.common.callbacks.BaseCallback):
        def __init__(self):
            super(CallBack, self).__init__()
            self.last_log = 0
            task_idx_mask = torch.div(torch.arange(hyper.num_envs), hyper.num_envs_per_task, rounding_mode="trunc")
            self.task_idx_mask = task_idx_mask.unsqueeze(0)

        def _on_step(self):
            meta_policy = self.locals["self"].policy.meta_policy
            obs_tensor = self.locals["obs_tensor"]
            features = meta_policy.extract_features(obs_tensor)
            latent_pi, latent_vf = meta_policy.mlp_extractor(features)
            task_idx_mask = self.task_idx_mask.squeeze()
            num_tasks = hyper.num_tasks
            distribution = meta_policy._get_action_dist_from_latent(latent_pi)
            for task_idx in range(num_tasks):
                task_dist = distribution.distribution.probs.detach()
                buddy_writer.add_histogram(f"task{task_idx}/meta_distr", task_dist[task_idx_mask == task_idx], self.num_timesteps)

            super(CallBack, self)._on_step()

        def _on_rollout_end(self):
            super(CallBack, self)._on_rollout_end()

            if (self.num_timesteps - self.last_log) <= hyper.log_iterate_every:
                return

            self.last_log = self.num_timesteps
            rollout_steps = self.locals["self"].n_steps
            num_tasks = hyper.num_tasks
            rollout_buffer = self.locals["rollout_buffer"]
            executed_options = rollout_buffer.current_options
            switches = rollout_buffer.current_options != rollout_buffer.previous_options
            switches[rollout_buffer.episode_starts.astype(bool)] = False

            buddy_writer.add_scalar("rollout/mean_returns", rollout_buffer.returns.mean(), self.num_timesteps)
            buddy_writer.add_scalar("rollout/change_points", switches.mean(), self.num_timesteps)

            if hasattr(rollout_buffer, "priority"):
                priority = rollout_buffer.priority
                if len(priority) < 3:
                    priority = priority.repeat(3)[:3]
                buddy_writer.add_histogram("train/priority", priority, self.num_timesteps)

            buddy_writer.add_histogram("rollout/executed_options", executed_options, self.num_timesteps)

            env_returns = rollout_buffer.returns.mean(0)
            task_returns = env_returns.reshape(num_tasks, -1).mean(1)

            task_idx_mask = self.task_idx_mask.repeat(rollout_steps, 1)
            for task_idx in range(num_tasks):
                task_executed_option = executed_options[task_idx_mask == task_idx]
                if len(task_executed_option) < 3:
                    task_executed_option = np.tile(task_executed_option, 3)[:3]
                buddy_writer.add_histogram(f"task{task_idx}/task_executed_options", task_executed_option, self.num_timesteps)

                if hasattr(rollout_buffer, "priority"):
                    buddy_writer.add_scalar(f"task{task_idx}/priority", rollout_buffer.priority[task_idx], self.num_timesteps)

                task_return = task_returns[task_idx]
                buddy_writer.add_scalar(f"task{task_idx}/mean_return", task_return, self.num_timesteps)

    cb = stable_baselines3.common.callbacks.CallbackList([
        option_baselines.common.callbacks.OptionRollout(envs, eval_freq=hyper.video_every, n_eval_episodes=5 if hyper.DEBUG else 1),
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


def wrap_envs():
    envs = stable_baselines3.common.vec_env.DummyVecEnv(
        env_fns=[functools.partial(task.make_env, hyper.num_envs, i) for i in range(hyper.num_envs)],
    )
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
    proc_num = 20
    # tb = experiment_buddy.deploy("mila", "sweep.yaml", proc_num=proc_num, run_per_agent=1, disabled=hyper.DEBUG, wandb_kwargs={'project': "ogw"}, )
    # tb = experiment_buddy.deploy("mila", "sweep.yaml", proc_num=25, disabled=hyper.DEBUG)
    # tb = experiment_buddy.deploy("mila", wandb_kwargs=wandb_kwargs)
    tb = experiment_buddy.deploy()
    main(tb, torch.device("cpu"))
