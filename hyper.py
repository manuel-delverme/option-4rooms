import sys

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

discount = 0.99
training_steps = 1_000_000

# Metrics
log_iterate_every = 100
slow_log_iterate_every = 10 * log_iterate_every
checkpoint_every = 100_000
video_every = 10_000
running_performance_window = 1_000

# Misc
seed = 0

# Good parameters
entropy_regularization = 0.01
switching_margin = 0.0
termination_regularization = 0.0

lr_pi = 7e-4

optimistic_init = True
task = "multitask"  # task = "door-key"

# Options
num_options = 1
num_envs_per_task = 6
num_tasks = 1

num_envs = num_envs_per_task * num_tasks
# option_schedule = np.array([0, 0])  # , 150_000])

# oversampling_rate = 1
