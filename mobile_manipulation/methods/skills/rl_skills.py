from collections import OrderedDict

import numpy as np
import torch
from gym import spaces
from habitat import Config, RLEnv
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import batch_obs

from mobile_manipulation.common.registry import (
    mm_registry as my_registry,
)
from mobile_manipulation.methods.skill import Skill
from mobile_manipulation.ppo.policy import ActorCritic
from mobile_manipulation.utils.common import get_state_dict_by_prefix


class RLSkill(Skill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.action_shape = self._action_space[self._config.ACTION].shape
        self.skill_index = self._config.get("SKILL_INDEX", "")
        self._init_policy(self._config.CKPT_PATH, self.skill_index)

    def _init_policy(self, ckpt_path: str, skill_idx: str = ""):
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint from {ckpt_path}")

        ckpt_config = ckpt_dict["config"]
        action_space = self._action_space[self._config.ACTION]
        policy_config = ckpt_config.RL["POLICY" + skill_idx]
        policy = baseline_registry.get_policy(policy_config.name)
        actor_critic: ActorCritic = policy.from_config(
            policy_config, self._obs_space, action_space
        )

        state_dict = ckpt_dict["state_dict" + skill_idx]
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
            value = outputs["value"]
            self._buffer.update(
                recurrent_hidden_states=outputs["rnn_hidden_states"],
                prev_actions=outputs["action"],
                masks=torch.ones(1, 1, device=self.device, dtype=torch.bool),
            )

        self._elapsed_steps += 1

        action = action.squeeze().cpu().numpy()
        value = value.item()
        step_action = {
            "action": self._config.ACTION,
            "action_args": action,
            "value": value,
        }
        return step_action

    def should_terminate(self, obs, **kwargs):
        return self.is_timeout()

    def to(self, device=None):
        self.device = device
        self.actor_critic.to(device)
        return self


@my_registry.register_skill
class PickRLSkill(RLSkill):
    def act(self, obs, **kwargs):
        step_action = super().act(obs, **kwargs)
        if self._config.get("DISABLE_RELEASE", True):
            # Hardcode, but should work
            is_grasped = obs["is_grasped"] > 0.5
            if is_grasped:
                gripper_action = step_action["action_args"][-1]
                step_action["action_args"][-1] = max(gripper_action, 0)

        return step_action

    def check_success_from_info(self):
        info = self._rl_env.habitat_env.get_metrics()
        success = info["rearrange_pick_success"]
        return success

    def check_success_from_obs(self, obs):
        info = self._rl_env.habitat_env.get_metrics()
        is_grasped = obs["is_grasped"] > 0.5
        if "gripper_to_resting_dist" in info:
            gripper_to_resting_dist = info["gripper_to_resting_dist"]
        else:
            gripper_pos = obs["gripper_pos_at_base"]
            resting_pos = obs["resting_pos_at_base"]
            gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        success = (
            is_grasped and gripper_to_resting_dist < self._config["THRESHOLD"]
        )
        return success

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        end_type = self._config.END_TYPE
        if end_type == "info":
            return self.check_success_from_info()
        elif end_type == "obs":
            return self.check_success_from_obs(obs)
        else:
            raise NotImplementedError(end_type)


@my_registry.register_skill
class NavRLSkill(RLSkill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        task = self._rl_env.habitat_env.task
        task_action = task.actions[self._config.ACTION]
        task_action.is_stop_called = False

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True

        end_type = self._config.END_TYPE
        if end_type == "gt":
            return self.check_success_gt()
        elif end_type == "non_stop":
            return False
        elif end_type == "call_stop":
            return self.check_stop_called()
        else:
            raise NotImplementedError(end_type)

    def check_success_gt(self):
        info = self._rl_env.get_info()
        success = info["rearrange_nav_success"]
        return success

    def check_stop_called(self):
        task = self._rl_env.habitat_env.task
        task_action = task.actions[self._config.ACTION]
        return task_action.is_stop_called


@my_registry.register_skill
class PlaceRLSkill(RLSkill):
    def act(self, obs, **kwargs):
        step_action = super().act(obs, **kwargs)
        if self._config.get("DISABLE_GRASP", True):
            # Hardcode, but should work
            is_grasped = obs["is_grasped"] > 0.5
            if not is_grasped:
                gripper_action = step_action["action_args"][-1]
                step_action["action_args"][-1] = min(gripper_action, 0)

        return step_action

    def check_success_from_info(self):
        info = self._rl_env.habitat_env.get_metrics()
        success = info["rearrange_place_success"]
        return success

    def check_success_from_obs(self, obs, **kwargs):
        info = self._rl_env.habitat_env.get_metrics()
        is_grasped = obs["is_grasped"] > 0.5
        if "gripper_to_resting_dist" in info:
            gripper_to_resting_dist = info["gripper_to_resting_dist"]
        else:
            gripper_pos = obs["gripper_pos_at_base"]
            resting_pos = obs["resting_pos_at_base"]
            gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        success = (
            not is_grasped
            and gripper_to_resting_dist < self._config["THRESHOLD"]
        )
        return success

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            # hardcode to release
            is_grasped = obs["is_grasped"] > 0.5
            if is_grasped:
                print("Release when timeout")
                self._rl_env.habitat_env.sim.gripper.desnap(False)
            return True

        end_type = self._config.END_TYPE
        if end_type == "info":
            return self.check_success_from_info()
        elif end_type == "obs":
            return self.check_success_from_obs(obs)
        else:
            raise NotImplementedError(end_type)


@my_registry.register_skill
class SetMarkerRLSkill(RLSkill):
    def reset(self, obs, **kwargs):
        self._has_set = False
        return super().reset(obs, **kwargs)

    def act(self, obs, **kwargs):
        if not self._has_set and self._get_gripper_distance(obs) > 0.3:
            self._has_set = True
        return super().act(obs, **kwargs)

    def _get_gripper_distance(self, obs):
        info = self._rl_env.habitat_env.get_metrics()
        if "gripper_to_resting_dist" in info:
            gripper_to_resting_dist = info["gripper_to_resting_dist"]
        else:
            gripper_pos = obs["gripper_pos_at_base"]
            resting_pos = obs["resting_pos_at_base"]
            gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        return gripper_to_resting_dist

    def check_success_from_obs(self, obs, **kwargs):
        is_grasped = obs["is_grasped"] > 0.5
        gripper_to_resting_dist = self._get_gripper_distance(obs)
        success = (
            not is_grasped
            and gripper_to_resting_dist <= self._config["THRESHOLD"]
        )
        return success and self._has_set

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            # hardcode to release
            is_grasped = obs["is_grasped"] > 0.5
            if is_grasped:
                print("Release when timeout")
                self._rl_env.habitat_env.sim.gripper.desnap(False)
            return True

        end_type = self._config.get("END_TYPE", "obs")
        if end_type == "obs":
            return self.check_success_from_obs(obs)
        else:
            raise NotImplementedError(end_type)


@my_registry.register_skill
class PickPlaceRLSkill(RLSkill):
    def reset(self, obs, **kwargs):
        return super().reset(obs, **kwargs)

    def act(self, obs, **kwargs):
        return super().act(obs, **kwargs)

    def _get_gripper_distance(self, obs):
        info = self._rl_env.habitat_env.get_metrics()
        if "gripper_to_resting_dist" in info:
            gripper_to_resting_dist = info["gripper_to_resting_dist"]
        else:
            gripper_pos = obs["gripper_pos_at_base"]
            resting_pos = obs["resting_pos_at_base"]
            gripper_to_resting_dist = np.linalg.norm(gripper_pos - resting_pos)
        return gripper_to_resting_dist

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            # hardcode to release
            is_grasped = obs["is_grasped"] > 0.5
            if is_grasped:
                print("Release when timeout")
                self._rl_env.habitat_env.sim.gripper.desnap(False)
            return True
        else:
            return False
