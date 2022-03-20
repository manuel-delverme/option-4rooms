import numpy as np
import stable_baselines3.common

import hyper


class CallBack(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super(CallBack, self).__init__()
        self.last_log = 0
        self.task_idx_mask = False
        self.option_observations = None
        self.option_policy = None

    def _on_step(self):
        super(CallBack, self)._on_step()

    def _on_rollout_end(self):
        super(CallBack, self)._on_rollout_end()

        if (self.num_timesteps - self.last_log) <= hyper.log_iterate_every:
            return
        print("progress", 1 - self.locals["self"]._current_progress_remaining, self.num_timesteps)
        self.last_log = self.num_timesteps
        rollout_buffer = self.locals["rollout_buffer"]
        env = self.locals["env"]
        agent = self.locals["self"]

        executed_options = rollout_buffer.current_options
        switches = rollout_buffer.current_options != rollout_buffer.previous_options
        switches[rollout_buffer.episode_starts.astype(bool)] = False
        ref_env = env.envs[0]

        if self.option_observations is None:
            self.option_observations = np.zeros((agent.num_options, ref_env.num_states))
        if self.option_policy is None:
            self.option_policy = np.zeros((agent.num_options, ref_env.num_states, ref_env.num_actions))

        for opt_idx in range(agent.num_options):
            opt_mask = (rollout_buffer.current_options == opt_idx)
            opt_obs = rollout_buffer.observations['state'][opt_mask]
            opt_state = rollout_buffer.observations['state'][opt_mask].argmax(axis=1)
            opt_acts = rollout_buffer.actions[opt_mask].squeeze(1).astype(np.int32)
            opt_acts_onehot = np.eye(ref_env.num_actions)[opt_acts]

            self.option_observations[opt_idx] += opt_obs.sum(0)
            self.option_policy[opt_idx][opt_state] += opt_acts_onehot

        if self.num_timesteps % hyper.slow_log_iterate_every == 0 and self.num_timesteps:
            for opt_idx in range(agent.num_options):
                norm_map = self.option_observations[opt_idx] / self.option_observations[opt_idx].sum()
                norm_policy = self.option_policy[opt_idx] / self.option_policy[opt_idx].sum(1, keepdims=True)
                norm_policy[np.isnan(norm_policy)] = 0
                self.logger.plot(*ref_env.plot_s(f"options/occupancy{opt_idx}", norm_map, vmin=0, vmax=0.5), self.num_timesteps)
                self.logger.plot(*ref_env.plot_sa(f"options/policy{opt_idx}", norm_policy), self.num_timesteps)
                self.logger.add_scalar(f"options/occupancy{opt_idx}_norm", np.linalg.norm(norm_map), self.num_timesteps)
            self.option_observations.fill(0)
            self.option_policy.fill(0)

        self.logger.add_scalar("rollout/mean_returns", rollout_buffer.returns.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/mean_rewards", rollout_buffer.rewards.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/change_points", switches.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/action_gap", np.inf, self.num_timesteps)
        self.logger.add_histogram("rollout/executed_options", executed_options, self.num_timesteps)
