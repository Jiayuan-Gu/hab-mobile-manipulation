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
_C.BASE_RUN_DIR = "data/results/rearrange/composite/{fileName}"
_C.BASE_TASK_CONFIG_PATH = ""
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.ENV_NAME = ""
_C.TORCH_GPU_ID = 0
# overwrite the simulator config, refer to construct_envs
_C.SIMULATOR_GPU_ID = 0
_C.SENSORS = []

_C.LOG_INTERVAL = 10
_C.LOG_FILE = "{baseRunDir}/{prefix}/log.txt"

# Video
_C.VIDEO_DIR = "{baseRunDir}/{prefix}/video"

_C.VERBOSE = True
_C.DEBUG = False
_C.PREFIX = ""
# NOTE(jigu): The default has been changed to True!
_C.FORCE_TORCH_SINGLE_THREADED = True

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
# Since RLEnv might provide multiple action spaces,
# we need to specify one action space for RL training.
_C.RL.ACTION_NAME = ""
_C.RL.REWARD_MEASURES = []
_C.RL.SUCCESS_MEASURE = ""
_C.RL.SUCCESS_REWARD = 0.0
_C.RL.SLACK_REWARD = 0.0

# -------------------------------------------------------------------------- #
# Composite policy
# -------------------------------------------------------------------------- #
_C.SOLUTION = CN()
_C.SOLUTION.SKILLS = []
_C.SOLUTION.Terminate = CN()
_C.SOLUTION.Terminate.TYPE = "Terminate"
