#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Type, Union

import gym
from habitat import Config, Env, RLEnv, make_dataset
from .vector_env import VectorEnv, ThreadedVectorEnv


def make_env_fn(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    wrappers: List[gym.Wrapper],
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def construct_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    split_dataset: bool,
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    wrappers: List[gym.Wrapper] = (),
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_environments as well as information
            necessary to create individual environments.
        env_class: class type of the envs to be created.
        split_dataset: whether to split datasets according to scene_ids
        workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
        auto_reset_done: Passed to :ref:`habitat.VectorEnv`'s constructor
        wrappers: gym wrappers for env_class

    Returns:
        VectorEnv object created according to specification.
    """

    num_envs = config.NUM_ENVIRONMENTS
    configs = []
    env_classes = [env_class] * num_envs

    # NOTE(jigu): One scene per process can maximize the simulation speed.
    if split_dataset:
        dataset = make_dataset(
            config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
        )
        # print(len(dataset.scene_ids))
        datasets = dataset.get_splits(
            num_envs, sort_by_episode_id=True, allow_uneven_splits=True
        )
        episode_splits = [x.episode_ids for x in datasets]
        # for dataset in datasets:
        #     print(dataset.num_episodes)
        #     print(dataset.scene_ids)

    # Prepare the config for each environment
    for i in range(num_envs):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if split_dataset:
            task_config.DATASET.EPISODE_IDS = episode_splits[i]

        # NOTE(jigu): overwrite here to avoid polluating config saved in ckpt
        # overwrite simulator config
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    # Vectorize environments
    debug = os.environ.get("HABITAT_ENV_DEBUG", 0)
    vec_env_cls = ThreadedVectorEnv if debug else VectorEnv
    envs = vec_env_cls(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes, [wrappers] * num_envs)),
        workers_ignore_signals=workers_ignore_signals,
        auto_reset_done=auto_reset_done,
    )
    return envs
