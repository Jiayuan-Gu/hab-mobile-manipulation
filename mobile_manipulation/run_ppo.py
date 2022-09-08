#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
import random
import time

import numpy as np
import torch
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions.tasks.rearrange
import mobile_manipulation.ppo
from mobile_manipulation.config import get_config
from mobile_manipulation.utils.common import get_run_name, warn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--cfg",
        dest="config_path",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", type=str, help="dataset split")
    parser.add_argument(
        "--no-tb", action="store_true", help="disable tensorboard"
    )
    parser.add_argument(
        "--no-video", action="store_true", help="disable video"
    )

    args = parser.parse_args()
    config = get_config(args.config_path, args.opts)
    preprocess_config(config, args)
    execute_exp(config, args.run_type)


def preprocess_config(config: Config, args):
    config_path = args.config_path
    run_type = args.run_type

    config.defrost()

    # placeholders supported in config
    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    timestamp = time.strftime("%y%m%d")
    substitutes = dict(
        fileName=fileName,
        runName=runName,
        runType=run_type,
        timestamp=timestamp,
    )

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["CHECKPOINT_FOLDER"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )

    for key in ["LOG_FILE", "TENSORBOARD_DIR", "VIDEO_DIR"]:
        if key not in config:
            warn(f"'{key}' is missed in the config")
            continue
        if run_type == "train":
            prefix = config.PREFIX
        else:
            prefix = config.EVAL.PREFIX or config.PREFIX
        config[key] = config[key].format(
            prefix=prefix,
            baseRunDir=config.BASE_RUN_DIR,
            **substitutes,
        )

    # Support relative path like "@/ckpt.pth"
    config.EVAL.CKPT_PATH = config.EVAL.CKPT_PATH.replace(
        "@", config.CHECKPOINT_FOLDER
    )

    # Override
    if args.split is not None:
        if run_type == "train":
            config.TASK_CONFIG.DATASET.SPLIT = args.split
        else:
            config.EVAL.SPLIT = args.split
    if args.no_tb:
        config.TENSORBOARD_DIR = ""
    if args.no_video:
        config.VIDEO_OPTION = []

    if args.debug:
        if run_type == "train":
            config.LOG_FILE = ""
            config.CHECKPOINT_FOLDER = ""
            config.TENSORBOARD_DIR = ""
            config.LOG_INTERVAL = 1
        else:
            config.LOG_FILE = ""
            config.TENSORBOARD_DIR = ""
            config.VIDEO_OPTION = []

    config.freeze()


def execute_exp(config: Config, run_type: str) -> None:
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_cls = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_cls is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_cls(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    else:
        raise NotImplementedError(run_type)


if __name__ == "__main__":
    main()
