import argparse
import json
import os
import os.path as osp
import re

import magnum as mn
import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.utils.common import batch_obs, generate_video

import mobile_manipulation.methods.skills
from habitat_extensions.tasks.rearrange import RearrangeRLEnv
from habitat_extensions.tasks.rearrange.play import get_action_from_key
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat_extensions.utils.visualizations.utils import put_info_on_image
from mobile_manipulation.config import get_config
from mobile_manipulation.methods.skill import CompositeSkill
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_git_commit_id,
    get_run_name,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1


def preprocess_config(config_path: str, config: Config):
    config.defrost()

    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    substitutes = dict(fileName=fileName, runName=runName)

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["LOG_FILE", "VIDEO_DIR"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )


def update_ckpt_path(config: Config, seed: int):
    config.defrost()
    for k in config:
        if k == "CKPT_PATH":
            ckpt_path = config[k]
            new_ckpt_path = re.sub(r"seed=[0-9]+", f"seed={seed}", ckpt_path)
            print(f"Update {ckpt_path} to {new_ckpt_path}")
            config[k] = new_ckpt_path
        elif isinstance(config[k], Config):
            update_ckpt_path(config[k], seed)
    config.freeze()


def update_sensor_resolution(config: Config, height, width):
    config.defrost()
    sensor_names = [
        "THIRD_RGB_SENSOR",
        "RGB_SENSOR",
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
    ]
    for name in sensor_names:
        sensor_cfg = config.TASK_CONFIG.SIMULATOR[name]
        sensor_cfg.HEIGHT = height
        sensor_cfg.WIDTH = width
        print(f"Update {name} resolution")
    config.freeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="config_path", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    # Episodes
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to shuffle test episodes",
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes to evaluate"
    )
    parser.add_argument(
        "--episode-ids", type=str, help="episodes ids to evaluate"
    )

    # Save
    parser.add_argument("--save-video", choices=["all", "failure"])
    parser.add_argument("--save-log", action="store_true")

    # Viewer
    parser.add_argument(
        "--viewer", action="store_true", help="enable OpenCV viewer"
    )
    parser.add_argument("--viewer-delay", type=int, default=10)
    parser.add_argument(
        "--play", action="store_true", help="enable input control"
    )

    # Policy
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-seed", type=int)

    # Rendering
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--render-info", action="store_true")
    parser.add_argument(
        "--no-rgb", action="store_true", help="disable rgb observations"
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="use high resolution for visualization",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Configure
    # ---------------------------------------------------------------------------- #
    config = get_config(args.config_path, opts=args.opts)
    preprocess_config(args.config_path, config)
    torch.set_num_threads(1)

    config.defrost()
    if args.split is not None:
        config.TASK_CONFIG.DATASET.SPLIT = args.split
    if not args.shuffle:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    if args.no_rgb:
        sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
            x for x in sensors if "RGB" not in x
        ]
    config.freeze()

    if args.train_seed is not None:
        update_ckpt_path(config, seed=args.train_seed)

    if args.high_res:
        update_sensor_resolution(config, height=720, width=1080)

    if args.save_log:
        if config.LOG_FILE:
            log_dir = os.path.dirname(config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(config.LOG_FILE)
        logger.info(config)
        logger.info("commit id: {}".format(get_git_commit_id()))

    # For reproducibility, just skip other episodes
    if args.episode_ids is not None:
        eval_episode_ids = eval(args.episode_ids)
        eval_episode_ids = [str(x) for x in eval_episode_ids]
    else:
        eval_episode_ids = None

    # ---------------------------------------------------------------------------- #
    # Initialize env
    # ---------------------------------------------------------------------------- #
    env = RearrangeRLEnv(config)
    env = HabitatActionWrapperV1(env)
    env.seed(config.TASK_CONFIG.SEED)
    print("obs space", env.observation_space)
    print("action space", env.action_space)

    # -------------------------------------------------------------------------- #
    # Initialize policy
    # -------------------------------------------------------------------------- #
    policy = CompositeSkill(config.SOLUTION, env)
    policy.to(args.device)

    # -------------------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------------------- #
    num_episodes = env.number_of_episodes
    # num_episodes = len(env.habitat_env.episode_iterator.episodes)
    if args.num_episodes is not None:
        num_episodes = args.num_episodes

    done, info = True, {}
    all_episode_stats = []
    episode_reward = 0
    failure_episodes = []

    if args.save_video is not None:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    rgb_frames = []

    if args.viewer:
        viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)

    for i_ep in range(num_episodes):
        ob = env.reset()
        policy.reset(ob)

        episode_reward = 0.0
        info = {}
        rgb_frames = []
        episode_id = env.current_episode.episode_id
        scene_id = env.current_episode.scene_id

        # Skip episode and keep reproducibility
        if eval_episode_ids is not None and episode_id not in eval_episode_ids:
            print("Skip episode", episode_id)
            continue

        while True:
            step_action = policy.act(ob)
            if step_action is None:
                print("Terminate the episode given none action")
                break

            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.viewer or args.save_video:
                # Add additional info
                info["values"] = step_action.get("values")
                info["value"] = step_action.get("value")
                info["success_probs"] = step_action.get("success_probs")

                metrics = extract_scalars_from_info(info)
                if args.render_mode == "human":
                    frame = env.render(
                        "human",
                        info=metrics,
                        overlay_info=False,
                        show_info=args.render_info,
                    )
                else:
                    frame = env.render(args.render_mode)
                    if args.render_info:
                        frame = put_info_on_image(
                            frame, info=metrics, overlay=False
                        )
                rgb_frames.append(frame)

            if args.viewer:
                key = viewer.imshow(
                    frame[..., :3], delay=0 if args.play else args.viewer_delay
                )

            if args.play:
                play_action = get_action_from_key(key, "BaseArmGripperAction")
                if play_action is not None:
                    step_action = play_action
            # -------------------------------------------------------------------------- #

            ob, reward, done, info = env.step(step_action)
            episode_reward += reward
            if args.viewer and key == "r":
                done = True
            if done:
                break

        # -------------------------------------------------------------------------- #
        # Update stats
        # -------------------------------------------------------------------------- #
        metrics = extract_scalars_from_info(info)
        episode_stats = metrics.copy()
        episode_stats["return"] = episode_reward
        all_episode_stats.append(episode_stats)

        logger.info(
            "Episode {} ({}/{}): {}".format(
                episode_id, i_ep, num_episodes, episode_stats
            )
        )

        success = metrics.get(config.RL.SUCCESS_MEASURE, -1)
        is_failure = success == False

        if args.save_video == "all" or (
            args.save_video == "failure" and is_failure
        ):
            generate_video(
                video_option=["disk"],
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=episode_id,
                checkpoint_idx=-1,
                metrics={"success": success},
                fps=30,
                tb_writer=None,
            )

        if is_failure:
            failure_episodes.append(episode_id)

        if eval_episode_ids is not None:
            if len(all_episode_stats) >= len(eval_episode_ids):
                print("Completed")
                break

    env.close()

    # logging metrics
    aggregated_stats = {
        k: np.mean([ep_info[k] for ep_info in all_episode_stats])
        for k in all_episode_stats[0].keys()
    }
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.4f}")

    failure_episodes = sorted(failure_episodes)
    failure_episodes_str = ",".join(map(str, failure_episodes))
    logger.info("Failure episodes:\n{}".format(failure_episodes_str))

    if args.save_log:
        json_path = config.LOG_FILE.replace("log.txt", "result.json")
        with open(json_path, "w") as f:
            json.dump(all_episode_stats, f, indent=2)


if __name__ == "__main__":
    main()
