import copy
from typing import List, Optional

import magnum as mn
import numpy as np
from habitat import Config, Dataset, RLEnv
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.utils.visualizations.utils import (
    draw_border,
    observations_to_image,
    put_info_on_image,
)

# isort: off
from .sim import RearrangeSim
from .task import RearrangeTask
from . import actions, sensors
from . import sub_tasks, composite_tasks, composite_sensors
from .sensors import GripperStatus


@baseline_registry.register_env(name="RearrangeRLEnv-v0")
class RearrangeRLEnv(RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._prev_env_obs = None

        super().__init__(self._core_env_config, dataset=dataset)

    def reset(self):
        observations = super().reset()
        self._prev_env_obs = observations
        # self._prev_env_obs = copy.deepcopy(observations)
        return observations

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        self._prev_env_obs = observations
        return observations, reward, done, info

    def get_success(self):
        measures = self._env.task.measurements.measures
        success_measure = self._rl_config.SUCCESS_MEASURE
        if success_measure in measures:
            success = measures[success_measure].get_metric()
        else:
            success = False
        if self._rl_config.get("SUCCESS_ON_STOP", False):
            success = success and self._env.task.should_terminate
        return success

    def get_reward(self, observations: Observations):
        metrics = self._env.get_metrics()

        reward = self._rl_config.SLACK_REWARD
        for reward_measure in self._rl_config.REWARD_MEASURES:
            # print(reward_measure, metrics[reward_measure])
            reward += metrics[reward_measure]

        if self.get_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def get_done(self, observations: Observations):
        # NOTE(jigu): episode is over when task.is_episode_active is False,
        # or time limit is passed.
        done = self._env.episode_over

        success = self.get_success()
        end_on_success = self._rl_config.get("END_ON_SUCCESS", True)
        if success and end_on_success:
            done = True

        return done

    def get_info(self, observations: Observations):
        info = self._env.get_metrics()
        info["is_episode_active"] = self._env.task.is_episode_active
        if self._env.task.is_episode_active:
            # The episode can only be truncated if not active
            assert (
                not self._env.task.is_episode_truncated
            ), self._env._elapsed_steps
            info["is_episode_truncated"] = self._env._past_limit()
        else:
            info["is_episode_truncated"] = self._env.task.is_episode_truncated
        info["elapsed_steps"] = self._env._elapsed_steps
        return info

    def get_reward_range(self):
        # Have not found its usage, but required to be implemented.
        return (np.finfo(np.float32).min, np.finfo(np.float32).max)

    def render(self, mode: str = "human", **kwargs) -> np.ndarray:
        if mode == "human":
            obs = self._prev_env_obs.copy()
            info = kwargs.get("info", {})
            show_info = kwargs.get("show_info", True)
            overlay_info = kwargs.get("overlay_info", True)
            render_uuid = kwargs.get("render_uuid", "robot_third_rgb")

            # rendered_frame = self._env.sim.render(render_uuid)
            rendered_frame = self._env.task.render(render_uuid)
            # rendered_frame = obs[render_uuid]

            # gripper status
            measures = self._env.task.measurements.measures
            gripper_status = measures.get("gripper_status", None)
            if gripper_status is None:
                gripper_status = measures.get("gripper_status_v1", None)
            if gripper_status is not None:
                if gripper_status.status == GripperStatus.PICK_CORRECT:
                    rendered_frame = draw_border(
                        rendered_frame, (0, 255, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.PICK_WRONG:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 0, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.DROP:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 255, 0), alpha=0.5
                    )

            if show_info:
                rendered_frame = put_info_on_image(
                    rendered_frame, info, overlay=overlay_info
                )
            obs[render_uuid] = rendered_frame

            return observations_to_image(obs)
        else:
            return super().render(mode=mode)
