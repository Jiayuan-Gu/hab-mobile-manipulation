import argparse
import os
from typing import Dict
import cv2

import habitat
import magnum as mn
import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs

from habitat_extensions.robots.pybullet_utils import PybulletRobot, pose2mat
from mobile_manipulation.methods.skill import Skill
from mobile_manipulation.ppo.policy import ActorCritic
from mobile_manipulation.utils.common import get_state_dict_by_prefix


class Skill:
    def __init__(self, timeout=0, **kwargs):
        self.timeout = timeout

    def reset(self, obs, **kwargs):
        self._elapsed_steps = 0

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def should_terminate(self, obs, **kwargs):
        raise NotImplementedError

    def is_timeout(self):
        if self.timeout > 0:
            return self._elapsed_steps >= self.timeout
        else:
            return False

    def to(self, device):
        return self


class ResetArm(Skill):
    resting_qpos = np.float32([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])

    def act(self, obs, **kwargs):
        cur_qpos = obs["arm_joint_pos"]
        diff_qpos = self.resting_qpos - cur_qpos
        delta_qpos = np.clip(diff_qpos, -0.0125, 0.0125)
        delta_qpos[np.abs(delta_qpos) < 0.005] = 0

        self._elapsed_steps += 1

        return {
            "action": "ARM_ACTION",
            "action_args": {
                "arm_action": delta_qpos / 0.0125,
                "grip_action": 1 if obs["is_grasped"] else -1,
            },
        }

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True

        cur_qpos = obs["arm_joint_pos"]
        diff_qpos = self.resting_qpos - cur_qpos
        # print(diff_qpos)
        return np.max(np.abs(diff_qpos)) < 0.02


class RLSkill(Skill):
    def __init__(self, obs_space, ckpt_path, **kwargs):
        super().__init__(**kwargs)

        self.obs_space = obs_space
        self.action_space = self.get_action_space()
        self.action_shape = self.action_space.shape

        self.ckpt_path = ckpt_path
        self._init_policy(ckpt_path)
        self.device = None

    def get_action_space(self) -> spaces.Space:
        raise NotImplementedError

    def _init_policy(self, ckpt_path: str):
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint from {ckpt_path}")

        ckpt_config = ckpt_dict["config"]
        policy_config = ckpt_config.RL["POLICY"]
        policy = baseline_registry.get_policy(policy_config.name)
        actor_critic: ActorCritic = policy.from_config(
            policy_config, self.obs_space, self.action_space
        )

        state_dict = ckpt_dict["state_dict"]
        state_dict = get_state_dict_by_prefix(state_dict, "actor_critic.")
        actor_critic.load_state_dict(state_dict)
        actor_critic.eval()

        self.actor_critic = actor_critic
        self._ckpt_config = ckpt_config

        # cache for convenience
        self.num_recurrent_layers = self.actor_critic.net.num_recurrent_layers
        self.rnn_hidden_size = self.actor_critic.net.rnn_hidden_size

    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)

        self._buffer = dict(
            recurrent_hidden_states=torch.zeros(
                1, self.num_recurrent_layers, self.rnn_hidden_size
            ),
            prev_actions=torch.zeros(1, *self.action_shape),
            masks=torch.zeros(1, 1, dtype=torch.bool),
        )
        self._buffer = {k: v.to(self.device) for k, v in self._buffer.items()}

    def act(self, obs, **kwargs):
        with torch.no_grad():
            batch = batch_obs([obs], device=self.device)
            step_batch = dict(observations=batch, **self._buffer)
            outputs = self.actor_critic.act(step_batch, deterministic=True)
            action = outputs["action"]
            # value = outputs["value"]
            self._buffer.update(
                recurrent_hidden_states=outputs["rnn_hidden_states"],
                prev_actions=outputs["action"],
                masks=torch.ones(1, 1, device=self.device, dtype=torch.bool),
            )

        self._elapsed_steps += 1

        action = action.squeeze().cpu().numpy()
        # value = value.item()
        step_action = self.parse_raw_action(action)
        return step_action

    def parse_raw_action(self, action: np.ndarray) -> Dict:
        raise NotImplementedError

    def should_terminate(self, obs, **kwargs):
        return self.is_timeout()

    def to(self, device=None):
        self.device = device
        self.actor_critic.to(device)
        return self


class NavRLSkill(RLSkill):
    possible_velocities = np.array(
        [
            [lin_vel, ang_vel]
            for lin_vel in np.linspace(-0.5, 1.0, 4)
            for ang_vel in np.linspace(-1.0, 1.0, 5)
        ]
    )

    def get_action_space(self):
        return spaces.Discrete(4 * 5)

    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        self.is_stop_called = False

    def parse_raw_action(self, action):
        action = action.item()
        assert isinstance(action, int), action
        velocity = self.possible_velocities[action]
        # print("velocity", velocity, action)

        if np.allclose(velocity, 0):
            self.is_stop_called = True
        else:
            self.is_stop_called = False

        return {
            "action": "BASE_VELOCITY",
            "action_args": {"base_vel": velocity},
        }

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self.is_stop_called


class PickRLSkill(RLSkill):
    def get_action_space(self):
        return spaces.Box(-1, 1, [8])

    def parse_raw_action(self, action):
        return {
            "action": "ARM_ACTION",
            "action_args": {
                "arm_action": action[:7],
                "grip_action": action[7:],
            },
        }

    def act(self, obs, **kwargs):
        step_action = super().act(obs, **kwargs)
        if obs["is_grasped"] > 0.5:
            step_action["action_args"]["grip_action"] = 1.0
        return step_action

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        if obs["is_grasped"] < 0.5:
            return False
        gripper_pos = obs["gripper_pos_at_base"]
        resting_pos = np.float32([0.5, 1.0, 0.0])
        gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        return gripper_to_resting_dist <= 0.15


class PlaceRLSkill(RLSkill):
    def get_action_space(self):
        return spaces.Box(-1, 1, [8])

    def parse_raw_action(self, action):
        return {
            "action": "ARM_ACTION",
            "action_args": {
                "arm_action": action[:7],
                "grip_action": action[7:],
            },
        }

    def act(self, obs, **kwargs):
        step_action = super().act(obs, **kwargs)
        if obs["is_grasped"] < 0.5:
            step_action["action_args"]["grip_action"] = -1.0
        return step_action

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        if obs["is_grasped"] > 0.5:
            return False
        gripper_pos = obs["gripper_pos_at_base"]
        resting_pos = np.float32([0.5, 1.0, 0.0])
        gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        return gripper_to_resting_dist <= 0.15


class PPOAgent(habitat.Agent):
    def __init__(self):
        self._initialize_pyb()

        obs_space = spaces.Dict(
            robot_head_depth=spaces.Box(-1, 1, [128, 128, 1]),
            arm_joint_pos=spaces.Box(-1, 1, [7]),
            is_grasped=spaces.Box(-1, 1, [1]),
            gripper_pos_at_base=spaces.Box(-1, 1, [3]),
            pick_goal_at_gripper=spaces.Box(-1, 1, [3]),
            pick_goal_at_base=spaces.Box(-1, 1, [3]),
            place_goal_at_gripper=spaces.Box(-1, 1, [3]),
            place_goal_at_base=spaces.Box(-1, 1, [3]),
            nav_goal_at_base=spaces.Box(-1, 1, [3]),
        )

        # Individual skills
        self.nav_skill = NavRLSkill(
            obs_space=obs_space,
            ckpt_path="/home/jiayuan/projects/rearrange-challenge/data/results/rearrange/skills/challenge/nav_v0_disc_SCR/221007.seed=100.default/checkpoints/ckpt.-1.pth",
            timeout=500,
        ).to("cuda")
        self.pick_skill = PickRLSkill(
            obs_space=obs_space,
            ckpt_path="/home/jiayuan/projects/rearrange-challenge/data/results/rearrange/skills/challenge/pick_v0_joint_SCR/221007.seed=100.default/checkpoints/ckpt.-1.pth",
            timeout=200,
        ).to("cuda")
        self.place_skill = PlaceRLSkill(
            obs_space=obs_space,
            ckpt_path="/home/jiayuan/projects/rearrange-challenge/data/results/rearrange/skills/challenge/place_v0_joint_SCR/221007.seed=100.default/checkpoints/ckpt.-1.pth",
            timeout=200,
        ).to("cuda")
        self.reset_arm = ResetArm(timeout=50)

        # The sequence of skills to execute
        self.skill_seq = [
            self.nav_skill,
            self.reset_arm,
            self.pick_skill,
            self.reset_arm,
            self.nav_skill,
            self.reset_arm,
            self.place_skill,
            self.reset_arm,
        ]
        self.skill_idx = 0

    @property
    def current_skill(self):
        if self.skill_idx >= len(self.skill_seq):
            return None
        return self.skill_seq[self.skill_idx]

    def reset(self):
        self.skill_idx = 0
        self.current_skill.reset(None)

    def _initialize_pyb(self):
        ARM_URDF = "/home/jiayuan/projects/rearrange-challenge/habitat_extensions/assets/robots/hab_fetch/robots/hab_fetch_arm_v2.urdf"
        self.pyb_robot = PybulletRobot(
            ARM_URDF, joint_indices=[0, 1, 2, 3, 4, 5, 6], ee_link_idx=8
        )
        self.hab2pyb = mn.Matrix4(
            np.float32(
                [
                    [1, 0, 0, -0.0036],
                    [0, 0.0107961, -0.9999417, 0],
                    [0, 0.9999417, 0.0107961, 0.0014],
                    [0, 0, 0, 1],
                ],
            )
        )
        self.pyb2hab = self.hab2pyb.inverted()

    def parse_raw_obs(self, raw_obs: dict):
        # Resize depth
        robot_head_depth = raw_obs["robot_head_depth"]
        # robot_head_depth = cv2.resize(
        #     robot_head_depth, (128, 128), cv2.INTER_NEAREST
        # )[..., None]

        # Compute end-effector pose
        arm_joint_pos = raw_obs["joint"]
        self.pyb_robot.set_joint_states(arm_joint_pos)
        ee_pos = self.pyb_robot.ee_state[0]
        ee_quat = self.pyb_robot.ee_state[1]
        ee_T_at_pyb_base = mn.Matrix4(pose2mat(ee_pos, ee_quat))
        # print("ee_T_at_pyb_base", ee_T_at_pyb_base)
        ee_T_at_hab_base = self.pyb2hab @ ee_T_at_pyb_base
        gripper_pos_at_base = ee_T_at_hab_base.translation
        # print(gripper_pos_at_base)
        # print("ee_T_at_hab_base", ee_T_at_hab_base)
        ee_offset = mn.Matrix4.translation(mn.Vector3(0.08, 0, 0))
        # print("estimated ee_transform", ee_T_at_hab_base @ ee_offset)

        obj_start = mn.Vector3(raw_obs["obj_start_sensor"])
        pick_goal_at_gripper = ee_offset.transform_point(obj_start)
        pick_goal_at_base = ee_T_at_hab_base.transform_point(
            pick_goal_at_gripper
        )
        # print("pick_goal_at_base", pick_goal_at_base)

        obj_goal = mn.Vector3(raw_obs["obj_goal_sensor"])
        place_goal_at_gripper = ee_offset.transform_point(obj_goal)
        place_goal_at_base = ee_T_at_hab_base.transform_point(
            place_goal_at_gripper
        )
        # print("place_goal_at_base", place_goal_at_base)

        if raw_obs["is_holding"] > 0.5:
            nav_goal_at_base = place_goal_at_base
        else:
            nav_goal_at_base = pick_goal_at_base

        obs = dict(
            robot_head_depth=robot_head_depth,
            arm_joint_pos=arm_joint_pos,
            is_grasped=raw_obs["is_holding"],
            gripper_pos_at_base=np.float32(gripper_pos_at_base),
            pick_goal_at_gripper=np.float32(pick_goal_at_gripper),
            pick_goal_at_base=np.float32(pick_goal_at_base),
            place_goal_at_gripper=np.float32(place_goal_at_gripper),
            place_goal_at_base=np.float32(place_goal_at_base),
            nav_goal_at_base=np.float32(nav_goal_at_base),
        )
        return obs

    def act(self, obs):
        import cv2

        cv2.imshow("debug", obs["robot_head_rgb"][..., ::-1])
        cv2.waitKey(0)

        obs = self.parse_raw_obs(obs)

        if self.current_skill.should_terminate(obs):
            self.skill_idx += 1
            if self.current_skill is not None:
                skill_name = self.current_skill.__class__.__name__
                print("Skill <{}> begin.".format(skill_name))
                self.current_skill.reset(obs)

        if self.current_skill is None:
            step_action = {
                "action": "REARRANGE_STOP",
                "action_args": {"REARRANGE_STOP": [1.0]},
            }
            print("Terminate the episode")
        else:
            step_action = self.current_skill.act(obs)

        return step_action


def main():
    agent = PPOAgent()
    challenge = habitat.Challenge(eval_remote=False)
    challenge._env.seed(7)
    challenge.submit(agent)


if __name__ == "__main__":
    main()
