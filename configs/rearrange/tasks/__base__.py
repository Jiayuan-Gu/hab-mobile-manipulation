#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Important modification compared to the original habitat-sim default:
- ENABLE_PHYSICS=True
- RADIUS=0.3
- Default sensor resolution=(128, 128)
- SCENES_DIR=""
- SIMULATOR.SCENE=""
- ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS=-1
"""

from habitat import Config as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
cfg = _C
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
# _C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = ""
_C.TASK.ACTIONS = CN()
_C.TASK.POSSIBLE_ACTIONS = []
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True

# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
# _C.SIMULATOR.SCENE = (
#     "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
# )
_C.SIMULATOR.SCENE = ""
_C.SIMULATOR.SCENE_DATASET = "default"
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
# SIMULATOR_SENSOR.HEIGHT = 480
# SIMULATOR_SENSOR.WIDTH = 640
SIMULATOR_SENSOR.HEIGHT = 128
SIMULATOR_SENSOR.WIDTH = 128
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# CAMERA SENSOR
# -----------------------------------------------------------------------------
CAMERA_SIM_SENSOR = SIMULATOR_SENSOR.clone()
CAMERA_SIM_SENSOR.HFOV = 90  # horizontal field of view in degrees
CAMERA_SIM_SENSOR.SENSOR_SUBTYPE = "PINHOLE"

SIMULATOR_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
SIMULATOR_DEPTH_SENSOR.MIN_DEPTH = 0.0
SIMULATOR_DEPTH_SENSOR.MAX_DEPTH = 10.0
SIMULATOR_DEPTH_SENSOR.NORMALIZE_DEPTH = True

# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# ROBOT HEAD RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.HEAD_RGB_SENSOR.UUID = "robot_head_rgb"
# -----------------------------------------------------------------------------
# ROBOT HEAD DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.HEAD_DEPTH_SENSOR.UUID = "robot_head_depth"
# -----------------------------------------------------------------------------
# ROBOT HEAD SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_SEMANTIC_SENSOR = _C.SIMULATOR.SEMANTIC_SENSOR.clone()
_C.SIMULATOR.HEAD_SEMANTIC_SENSOR.UUID = "robot_head_semantic"
# -----------------------------------------------------------------------------
# ARM RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.ARM_RGB_SENSOR.UUID = "robot_arm_rgb"
# -----------------------------------------------------------------------------
# ARM DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.ARM_DEPTH_SENSOR.UUID = "robot_arm_depth"
# -----------------------------------------------------------------------------
# ARM SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_SEMANTIC_SENSOR = _C.SIMULATOR.SEMANTIC_SENSOR.clone()
_C.SIMULATOR.ARM_SEMANTIC_SENSOR.UUID = "robot_arm_semantic"
# -----------------------------------------------------------------------------
# 3rd RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.THIRD_RGB_SENSOR.UUID = "robot_third_rgb"
# -----------------------------------------------------------------------------
# 3rd DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.THIRD_DEPTH_SENSOR.UUID = "robot_third_depth"
# -----------------------------------------------------------------------------
# 3rd SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_SEMANTIC_SENSOR = _C.SIMULATOR.SEMANTIC_SENSOR.clone()
_C.SIMULATOR.THIRD_SEMANTIC_SENSOR.UUID = "robot_third_semantic"

# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
# _C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.RADIUS = 0.3
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# Whether or not the agent slides on collisions
_C.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
# _C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = False
_C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
_C.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = (
    "./data/default.physics_config.json"
)

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ""
_C.DATASET.SPLIT = "train"
# _C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.SCENES_DIR = ""
_C.DATASET.CONTENT_SCENES = ["*"]
# _C.DATASET.DATA_PATH = (
#     "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
# )
_C.DATASET.DATA_PATH = ""

# -----------------------------------------------------------------------------
