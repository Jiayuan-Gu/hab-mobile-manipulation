#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.config import Config as CN

_C = CN()
cfg = _C
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C.BASE_RUN_DIR = "data/results/rearrange/skills/{fileName}"
_C.BASE_TASK_CONFIG_PATH = ""
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = ""
_C.ENV_NAME = ""
_C.NUM_ENVIRONMENTS = 1
_C.register_deprecated_key("NUM_PROCESSES")
_C.TOTAL_NUM_STEPS = -1.0  # float for scientific notation
_C.TORCH_GPU_ID = 0

# overwrite the simulator config, refer to `construct_envs`
_C.SIMULATOR_GPU_ID = 0
_C.SENSORS = []

_C.PREFIX = ""
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "{baseRunDir}/{prefix}/log.{runType}.txt"
_C.CHECKPOINT_FOLDER = "{baseRunDir}/{prefix}/checkpoints"
_C.NUM_CHECKPOINTS = 10
# Number of model updates between checkpoints
# If set to -1, depends on NUM_CHECKPOINTS
_C.CHECKPOINT_INTERVAL = -1
_C.SUMMARIZE_INTERVAL = -1
_C.TENSORBOARD_DIR = "{baseRunDir}/{prefix}/tb"

_C.VERBOSE = True
_C.DEBUG = False
# NOTE(jigu): The default has been changed to True!
_C.FORCE_TORCH_SINGLE_THREADED = True
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.PREFIX = ""
# If emtpy, the latest checkpoint in CHECKPOINT_FOLDER will be used
_C.EVAL.CKPT_PATH = ""
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.DETERMINISTIC_ACTION = True
_C.EVAL.NUM_EPISODES = -1
_C.EVAL.BATCH_ENVS = False
# Video
_C.VIDEO_OPTION = ["disk"]  # ["disk", "tensorboard"]
_C.VIDEO_DIR = "{baseRunDir}/{prefix}/video"
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.ACTION_NAME = ""
_C.RL.REWARD_MEASURES = []
_C.RL.SUCCESS_MEASURE = ""
_C.RL.SUCCESS_REWARD = 0.0
_C.RL.SLACK_REWARD = 0.0
_C.RL.IGNORE_TRUNCATED = False  # for compatability
# -----------------------------------------------------------------------------
# POLICY CONFIG
# -----------------------------------------------------------------------------
_C.RL.POLICY = CN()
_C.RL.POLICY.name = ""
_C.RL.POLICY.actor_type = "gaussian"
_C.RL.POLICY.GAUSSIAN_ACTOR = CN()
_C.RL.POLICY.GAUSSIAN_ACTOR.hidden_sizes = []
_C.RL.POLICY.GAUSSIAN_ACTOR.action_activation = "tanh"
_C.RL.POLICY.GAUSSIAN_ACTOR.std_transform = "log"
_C.RL.POLICY.GAUSSIAN_ACTOR.min_std = -5
_C.RL.POLICY.GAUSSIAN_ACTOR.max_std = 2
_C.RL.POLICY.GAUSSIAN_ACTOR.conditioned_std = False
_C.RL.POLICY.GAUSSIAN_ACTOR.std_init_bias = 0.0
_C.RL.POLICY.CATEGORICAL_ACTOR = CN()
_C.RL.POLICY.CATEGORICAL_ACTOR.hidden_sizes = []
_C.RL.POLICY.CRITIC = CN()
_C.RL.POLICY.CRITIC.hidden_sizes = []
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 2
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 128
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
# NOTE(jigu): The default has been changed to True!
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.use_clipped_value_loss = True
_C.RL.PPO.use_recurrent_generator = True
_C.RL.PPO.reward_window_size = 100
