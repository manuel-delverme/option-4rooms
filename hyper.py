import sys

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

discount = 0.99
training_steps = 100_000

# Metrics
log_iterate_every = 100
slow_log_iterate_every = 10 * log_iterate_every
checkpoint_every = 100_000
video_every = 10_000
running_performance_window = video_every

# Misc
seed = 0

# Good parameters
entropy_regularization = 0.01
switching_margin = 0.01
termination_regularization = 0.00

lr_pi = 7e-4
lr_mu = 0.1

eps_decay = 100_000

optimistic_init = True
features = 256
task = "multitask"  # task = "door-key"
conv_channels = 32

# Options
num_options = 2
num_envs_per_task = 3
num_tasks = 4

num_envs = num_envs_per_task * num_tasks
# option_schedule = np.array([0, 0])  # , 150_000])

# oversampling_rate = 1
