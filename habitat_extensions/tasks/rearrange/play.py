import argparse

from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.tasks.rearrange.env import RearrangeRLEnv
from habitat_extensions.utils.viewer import OpenCVViewer
from mobile_manipulation.config import Config, get_config, load_config
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_flat_space_names,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1


def get_action_from_key(key, action_name):
    if "BaseArmGripperAction" in action_name:
        if key == "w":  # forward
            base_action = [1, 0]
        elif key == "s":  # backward
            base_action = [-1, 0]
        elif key == "a":  # turn left
            base_action = [0, 1]
        elif key == "d":  # turn right
            base_action = [0, -1]
        else:
            base_action = [0, 0]

        # End-effector is controlled
        if key == "i":
            arm_action = [1.0, 0.0, 0.0]
        elif key == "k":
            arm_action = [-1.0, 0.0, 0.0]
        elif key == "j":
            arm_action = [0.0, 1.0, 0.0]
        elif key == "l":
            arm_action = [0.0, -1.0, 0.0]
        elif key == "u":
            arm_action = [0.0, 0.0, 1.0]
        elif key == "o":
            arm_action = [0.0, 0.0, -1.0]
        else:
            arm_action = [0.0, 0.0, 0.0]

        if key == "f":  # grasp
            gripper_action = 1.0
        elif key == "g":  # release
            gripper_action = -1.0
        else:
            gripper_action = 0.0

        return {
            "action": "BaseArmGripperAction",
            "action_args": {
                "base_action": base_action,
                "arm_action": arm_action,
                "gripper_action": gripper_action,
            },
        }
    elif "ArmGripperAction" in action_name:
        if key == "i":
            arm_action = [1.0, 0.0, 0.0]
        elif key == "k":
            arm_action = [-1.0, 0.0, 0.0]
        elif key == "j":
            arm_action = [0.0, 1.0, 0.0]
        elif key == "l":
            arm_action = [0.0, -1.0, 0.0]
        elif key == "u":
            arm_action = [0.0, 0.0, 1.0]
        elif key == "o":
            arm_action = [0.0, 0.0, -1.0]
        else:
            arm_action = [0.0, 0.0, 0.0]

        if key == "f":
            gripper_action = 1.0
        elif key == "g":
            gripper_action = -1.0
        else:
            gripper_action = 0.0

        return {
            "action": "ArmGripperAction",
            "action_args": {
                "arm_action": arm_action,
                "gripper_action": gripper_action,
            },
        }
    elif action_name == "BaseVelAction":
        if key == "w":
            base_action = [1, 0]
        elif key == "s":
            base_action = [-1, 0]
        elif key == "a":
            base_action = [0, 1]
        elif key == "d":
            base_action = [0, -1]
        else:
            base_action = [0, 0]
        return {
            "action": "BaseVelAction",
            "action_args": {
                "velocity": base_action,
            },
        }
    elif action_name == "BaseVelAction2":
        if key == "w":
            base_action = [1, 0]
        elif key == "s":
            base_action = [-1, 0]
        elif key == "a":
            base_action = [0, 1]
        elif key == "d":
            base_action = [0, -1]
        else:
            base_action = [0, 0]
        if key == "z":
            stop = 1
        else:
            stop = 0
        return {
            "action": "BaseVelAction2",
            "action_args": {
                "velocity": base_action,
                "stop": stop,
            },
        }
    elif action_name == "BaseDiscVelAction":
        if key == "w":
            base_action = 17
        elif key == "s":
            base_action = 2
        elif key == "a":
            base_action = 9
        elif key == "d":
            base_action = 5
        elif key == "z":
            base_action = 7
        else:
            base_action = 17
        return {
            "action": "BaseDiscVelAction",
            "action_args": {
                "action": base_action,
            },
        }
    elif action_name == "EmptyAction":
        return {"action": "EmptyAction"}
    else:
        raise NotImplementedError(action_name)


def get_env_config_from_task_config(task_config: Config):
    config = Config()
    config.ENV_NAME = "RearrangeRLEnv-v0"
    config.RL = Config()
    config.RL.ACTION_NAME = task_config.TASK.POSSIBLE_ACTIONS[0]
    config.RL.REWARD_MEASURES = []
    config.RL.SUCCESS_MEASURE = ""
    config.RL.SUCCESS_REWARD = 0.0
    config.RL.SLACK_REWARD = 0.0
    config.TASK_CONFIG = task_config
    config.freeze()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="config_path",
        type=str,
        default="configs/rearrange/tasks/play.yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--action", type=str)
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="whether to sample an action from the action space",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to print verbose information",
    )
    parser.add_argument("--debug-obs", action="store_true")
    args = parser.parse_args()

    # -------------------------------------------------------------------------- #
    # Load config
    # -------------------------------------------------------------------------- #
    config = load_config(args.config_path)
    if "TASK_CONFIG" in config:
        # Reload as RLEnv
        config = get_config(args.config_path, opts=args.opts)
    else:
        config = get_env_config_from_task_config(config)
        if args.opts:
            config.defrost()
            config.merge_from_list(args.opts)
            config.freeze()

    # Override RL.ACTION_NAME
    if args.action:
        config.defrost()
        config.RL.ACTION_NAME = args.action
        config.freeze()

    # -------------------------------------------------------------------------- #
    # Env
    # -------------------------------------------------------------------------- #
    env_cls = baseline_registry.get_env(config.ENV_NAME)
    env: RearrangeRLEnv = env_cls(config)
    env = HabitatActionWrapperV1(env)
    print(config)
    print("obs_space", env.observation_space)
    print("action_space", env.action_space)
    state_keys = get_flat_space_names(env.observation_space)

    def reset():
        obs = env.reset()
        info = {}
        print("episode_id", env.habitat_env.current_episode.episode_id)
        print("scene_id", env.habitat_env.current_episode.scene_id)
        return obs, info

    env.seed(0)
    obs, info = reset()
    for k, v in obs.items():
        print(k, v.shape)
    viewer = OpenCVViewer(config.ENV_NAME)

    while True:
        metrics = extract_scalars_from_info(info)
        rendered_frame = env.render(info=metrics, overlay_info=False)
        key = viewer.imshow(rendered_frame)

        if key == "r":  # Press r to reset env
            obs, info = reset()
            continue

        if args.random_action:
            action = env.action_space.sample()
        else:
            # Please refer to this function for keyboard-action mapping
            action = get_action_from_key(key, config.RL.ACTION_NAME)

        obs, reward, done, info = env.step(action)
        if args.verbose:
            print("step", env.habitat_env._elapsed_steps)
            print("action", action)
            print("reward", reward)
            print("info", info)
        if args.debug_obs:
            print("obs", {k: v for k, v in obs.items() if k in state_keys})

        if done:
            print("Done")
            obs, info = reset()


if __name__ == "__main__":
    main()
