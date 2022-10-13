import magnum as mn
import numpy as np
from gym import spaces
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.tasks.rearrange.sim import RearrangeSim
from habitat_extensions.utils.geo_utils import wrap_angle

from ..sensors import MyMeasure, MySensor, PositionSensor
from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import compute_region_goals_v1
from .nav_task import RearrangeNavTask, RearrangeNavTaskV1


# -------------------------------------------------------------------------- #
# Sensor
# -------------------------------------------------------------------------- #
@registry.register_sensor
class BasePositionSensor(PositionSensor):
    cls_uuid = "base_pos"

    def _get_world_position(self, *args, **kwargs):
        return self._sim.robot.base_pos


@registry.register_sensor
class BaseHeadingSensor(MySensor):
    cls_uuid = "base_heading"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)

    def get_observation(self, *args, task: RearrangeTask, **kwargs):
        base_T = self._sim.robot.base_T
        if self.config.get("EPISODIC", True):
            base_T = task.start_base_T.inverted() @ base_T
        heading = base_T.transform_vector(mn.Vector3(1, 0, 0))
        return np.array([heading[0], heading[2]], dtype=np.float32)


@registry.register_sensor
class NavGoalSensor(PositionSensor):
    """Dynamic navigation goal according to whether the object is grasped."""

    cls_uuid = "nav_goal"

    def _get_world_position(self, *args, task: RearrangeNavTask, **kwargs):
        # if self._sim.gripper.grasped_obj == task.tgt_obj:
        if self._sim.gripper.is_grasped:
            return task.place_goal
        else:
            return task.pick_goal


# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
@registry.register_measure
class GeoDistanceToNavGoal(MyMeasure):
    cls_uuid = "geo_dist_to_nav_goal"

    def reset_metric(self, *args, episode: RearrangeEpisode, **kwargs):
        assert episode._shortest_path_cache is None, episode.episode_id
        return super().reset_metric(*args, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        start = self._sim.robot.base_pos
        goal_pos = task.nav_goal[0]
        # NOTE(jigu): a shortest path cache will be used if episode is passed.
        self._metric = self._sim.geodesic_distance(
            start, goal_pos, episode=episode
        )
        if np.isinf(self._metric):
            self._metric = 100.0


@registry.register_measure
class AngDistanceToNavGoal(MyMeasure):
    cls_uuid = "ang_dist_to_nav_goal"

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        start = self._sim.robot.base_ori
        goal_ori = task.nav_goal[1]
        self._metric = np.abs(wrap_angle(goal_ori - start))


@registry.register_measure
class RearrangeNavSuccess(MyMeasure):
    cls_uuid = "rearrange_nav_success"

    def reset_metric(self, *args, task: RearrangeNavTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GeoDistanceToNavGoal.cls_uuid, AngDistanceToNavGoal.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        measures = task.measurements.measures
        geo_dist = measures[GeoDistanceToNavGoal.cls_uuid].get_metric()
        ang_dist = measures[AngDistanceToNavGoal.cls_uuid].get_metric()
        self._metric = (
            geo_dist <= self._config.GEO_THRESHOLD
            and ang_dist <= self._config.ANG_THRESHOLD
        )

        # Deprecation: use "SUCCESS_ON_STOP" in RLEnv
        if self._config.get("ON_STOP", False):
            self._metric = self._metric and task._should_terminate


@registry.register_measure
class GeoDistanceToNavGoalsV1(MyMeasure):
    def reset_metric(self, *args, episode: RearrangeEpisode, **kwargs):
        assert episode._shortest_path_cache is None, episode.episode_id
        return super().reset_metric(*args, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTaskV1,
        episode: RearrangeEpisode,
        **kwargs
    ):
        start = self._sim.robot.base_pos
        # NOTE(jigu): a shortest path cache will be used if episode is passed.
        self._metric = self._sim.geodesic_distance(
            start, task.nav_goals, episode=episode
        )
        if self._metric <= self._config.get("MIN_DIST", -1.0):
            self._metric = 0.0


@registry.register_measure
class AngDistanceToGoal(MyMeasure):
    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        if task.sub_task == "place":
            goal_pos = task.place_goal
        else:
            goal_pos = task.pick_goal
        offset = goal_pos - np.array(self._sim.robot.base_pos)
        goal_ori = np.arctan2(-offset[2], offset[0])

        self._metric = np.abs(wrap_angle(goal_ori - self._sim.robot.base_ori))


@registry.register_measure
class AngDistanceToGoalV1(MyMeasure):
    def update_metric(self, *args, task: RearrangeNavTaskV1, **kwargs):
        offset = task.look_at_pos - np.array(self._sim.robot.base_pos)
        goal_ori = np.arctan2(-offset[2], offset[0])
        self._metric = np.abs(wrap_angle(goal_ori - self._sim.robot.base_ori))


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangeNavReward(MyMeasure):
    cls_uuid = "rearrange_nav_reward"

    def reset_metric(self, *args, task: RearrangeNavTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GeoDistanceToNavGoal.cls_uuid,
                AngDistanceToNavGoal.cls_uuid,
            ],
        )
        self.prev_geo_dist = 0.0
        self.prev_ang_dist = 0.0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        measures = task.measurements.measures
        geo_dist = measures[GeoDistanceToNavGoal.cls_uuid].get_metric()
        ang_dist = measures[AngDistanceToNavGoal.cls_uuid].get_metric()

        reward = 0.0

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        if geo_dist <= self._config.ANG_REWARD_THRESH:
            diff_ang_dist = self.prev_ang_dist - ang_dist
            diff_ang_dist = round(diff_ang_dist, 3)
            ang_dist_reward = diff_ang_dist * self._config.ANG_DIST_REWARD
            reward += ang_dist_reward

        self.prev_ang_dist = ang_dist

        diff_geo_dist = self.prev_geo_dist - geo_dist
        diff_geo_dist = round(diff_geo_dist, 3)
        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD
        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        self._metric = reward


# ---------------------------------------------------------------------------- #
# For composite rewards
# ---------------------------------------------------------------------------- #
@registry.register_measure
class RearrangeNavRewardV1(MyMeasure):
    cls_uuid = "rearrange_nav_reward"

    def reset_metric(
        self, *args, task: RearrangeTask, episode: RearrangeEpisode, **kwargs
    ):
        # assert episode._shortest_path_cache is None, episode.episode_id
        episode._shortest_path_cache = None

        if self._sim.gripper.is_grasped:
            T = mn.Matrix4.translation(task.place_goal)
        else:
            T = mn.Matrix4.translation(task.pick_goal)
        self.nav_goals = compute_region_goals_v1(
            self._sim,
            T,
            region=None,
            radius=self._config.RADIUS,
            height=self._sim.robot.base_pos[1],
        )

        self.prev_is_grasped = self._sim.gripper.is_grasped
        self.prev_geo_dist = None
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        reward = 0.0

        geo_dist = self._sim.geodesic_distance(
            self._sim.robot.base_pos, self.nav_goals, episode=episode
        )

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        if self.prev_geo_dist is None:
            diff_geo_dist = 0.0
        else:
            diff_geo_dist = self.prev_geo_dist - geo_dist
            diff_geo_dist = round(diff_geo_dist, 3)

        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD
        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        if self._sim.gripper.is_grasped != self.prev_is_grasped:
            reward -= self._config.GRASP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_GRASP", False
            )

        self._metric = reward
