import sys

import numpy as np

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

discount = 0.9
training_steps = 10_000_000

# Metrics
log_iterate_every = 100 if DEBUG else 1000  # 1000  # 5_000  # 10_000  # 5000
slow_log_iterate_every = 5 * log_iterate_every

# Misc
seed = 0

# Good parameters
entropy_regularization = 0.01
switching_margin = 0.01
termination_regularization = 0.00

lr_pi = 7e-4
lr_mu = 0.1

eps_decay = 100_000

running_performance_window = 500

optimistic_init = True
features = 256
task = "multitask"  # task = "door-key"
conv_channels = 32

checkpoint_every = slow_log_iterate_every * 25
video_every = slow_log_iterate_every * 5

# Options
num_options = 1
num_envs_per_task = 3
num_tasks = 2

num_envs = num_envs_per_task * num_tasks
option_schedule = np.array([0, 0])  # , 150_000])

# oversampling_rate = 1
