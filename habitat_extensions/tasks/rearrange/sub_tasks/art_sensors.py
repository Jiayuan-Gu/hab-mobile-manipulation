import magnum as mn
import numpy as np
from gym import spaces
from habitat.core.registry import registry

from ..sensors import (
    GripperStatus,
    GripperStatusMeasure,
    GripperToRestingDistance,
    MyMeasure,
    MySensor,
    PositionSensor,
)
from ..task import RearrangeTask
from .art_task import SetArticulatedObjectTask


# -------------------------------------------------------------------------- #
# Sensor
# -------------------------------------------------------------------------- #
@registry.register_sensor
class MarkerPositionSensor(PositionSensor):
    cls_uuid = "marker_pos"

    def _get_world_position(
        self, *args, task: SetArticulatedObjectTask, **kwargs
    ):
        return task.marker.pos


@registry.register_sensor
class MarkerJointPositionSensor(MySensor):
    cls_uuid = "marker_joint_pos"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def get_observation(self, *args, task: SetArticulatedObjectTask, **kwargs):
        return np.array([task.marker.qpos], dtype=np.float32)


@registry.register_sensor
class MarkerJointVelocitySensor(MySensor):
    cls_uuid = "marker_joint_vel"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def get_observation(self, *args, task: SetArticulatedObjectTask, **kwargs):
        return np.array([task.marker.qvel], dtype=np.float32)


# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
@registry.register_measure
class GripperToMarkerDistance(MyMeasure):
    cls_uuid = "gripper_to_marker_dist"

    def update_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        gripper_pos = self._sim.robot.gripper_pos
        marker_pos = task.marker.pos
        dist = np.linalg.norm(marker_pos - gripper_pos)
        self._metric = dist


@registry.register_measure
class MarkerToGoalDistance(MyMeasure):
    cls_uuid = "marker_to_goal_dist"

    def update_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        dist = task.tgt_qpos - task.marker.qpos
        if self._config.USE_ABS:
            dist = abs(dist)
        self._metric = dist


@registry.register_measure
class SetMarkerSuccess(MyMeasure):
    cls_uuid = "set_marker_success"

    def reset_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [MarkerToGoalDistance.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        measures = task.measurements.measures
        dist = measures[MarkerToGoalDistance.cls_uuid].get_metric()
        self._metric = dist <= self._config.THRESHOLD


@registry.register_measure
class RearrangeSetSuccess(MyMeasure):
    cls_uuid = "rearrange_set_success"

    def reset_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [SetMarkerSuccess.cls_uuid, GripperToRestingDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        measures = task.measurements.measures
        set_success = measures[SetMarkerSuccess.cls_uuid].get_metric()
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()
        rest_success = dist <= self._config.THRESHOLD
        is_grasped = self._sim.gripper.is_grasped
        self._metric = set_success and rest_success and (not is_grasped)


@registry.register_measure(name="GripperStatusV1")
class GripperStatusMeasureV1(GripperStatusMeasure):
    cls_uuid = "gripper_status_v1"

    def _update(self, *args, task: SetArticulatedObjectTask, **kwargs):
        curr_picked = self._sim.gripper.is_grasped
        if curr_picked:
            if self.status == GripperStatus.PICK_CORRECT:
                self.status = GripperStatus.HOLDING_CORRECT
            elif self.status == GripperStatus.PICK_WRONG:
                self.status = GripperStatus.HOLDING_WRONG
            elif self.status in [
                GripperStatus.NOT_HOLDING,
                GripperStatus.DROP,
            ]:
                if self._sim.gripper.grasped_marker_id == task.marker.uuid:
                    self.status = GripperStatus.PICK_CORRECT
                else:
                    self.status = GripperStatus.PICK_WRONG
                    # print("pick wrong", self._sim.gripper.grasped_marker_id)
                    # print("pick wrong", self._sim.gripper.grasped_obj_id)
        else:
            if self.status in [
                GripperStatus.PICK_CORRECT,
                GripperStatus.HOLDING_CORRECT,
                GripperStatus.PICK_WRONG,
                GripperStatus.HOLDING_WRONG,
            ]:
                self.status = GripperStatus.DROP
            else:
                self.status = GripperStatus.NOT_HOLDING


@registry.register_measure
class InRegion(MyMeasure):
    cls_uuid = "in_region"

    def reset_metric(self, *args, task: SetArticulatedObjectTask, **kwargs):
        self.allowed_region = mn.Range2D(*self._config.ALLOWED_REGION)
        if self._config.REF_ART_OBJ == "@marker":
            self.ref_art_obj = task.marker
        elif self._config.REF_ART_OBJ == "@marker_link":
            self.ref_art_obj = task.marker.link_node
        else:
            self.ref_art_obj = self._sim.art_objs.get(self._config.REF_ART_OBJ)
        self._T = self.ref_art_obj.transformation
        self._invT = self._T.inverted()
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, **kwargs):
        pos = self._invT.transform_point(self._sim.robot.base_T.translation)
        xz = mn.Vector2(pos[0], pos[2])
        self._metric = self.allowed_region.contains(xz)


@registry.register_measure
class OutOfRegionPenalty(MyMeasure):
    cls_uuid = "out_of_region_penalty"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [InRegion.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        in_region = measures[InRegion.cls_uuid].get_metric()

        if not in_region:
            self._metric = -self._config.PENALTY
            if self._config.END_EPISODE:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_EPISODE", False
                )
        else:
            self._metric = 0.0


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangeSetMarkerReward(MyMeasure):
    cls_uuid = "rearrange_set_reward"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    GripperToMarkerDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasureV1.cls_uuid,
                    SetMarkerSuccess.cls_uuid,
                ],
            )
        # It is fine to set zero as the reward is invalid when reset
        self.prev_dist_to_goal = None  # gripper
        self.prev_dist_to_goal2 = None  # marker
        self.prev_success = False
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures

        marker_to_goal_dist = measures[
            MarkerToGoalDistance.cls_uuid
        ].get_metric()
        gripper_to_marker_dist = measures[
            GripperToMarkerDistance.cls_uuid
        ].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()

        gs_measure = measures[GripperStatusMeasureV1.cls_uuid]
        gripper_status = gs_measure.status
        n_pick_correct = gs_measure.get_metric()["pick_correct"]
        n_drop = gs_measure.get_metric()["drop"]
        set_marker_success = measures[SetMarkerSuccess.cls_uuid].get_metric()
        # print("gripper_status", gripper_status)

        reward = 0.0

        if gripper_status == GripperStatus.PICK_CORRECT:
            if n_pick_correct == 1:  # first pick
                reward += self._config.PICK_REWARD

        if gripper_status == GripperStatus.PICK_WRONG:
            reward -= self._config.PICK_PENALTY
            if self._config.END_PICK_WRONG:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_PICK_WRONG", False
                )

        if gripper_status == GripperStatus.HOLDING_CORRECT:
            pass

        if gripper_status == GripperStatus.HOLDING_WRONG:
            raise RuntimeError

        if gripper_status == GripperStatus.DROP:
            if set_marker_success:
                if n_pick_correct > 0 and n_drop == 1:
                    reward += self._config.DROP_REWARD
            else:
                reward -= self._config.DROP_PENALTY
                if self._config.END_DROP:
                    task._is_episode_active = False
                    task._is_episode_truncated = self._config.get(
                        "TRUNCATE_DROP", False
                    )

        if gripper_status == GripperStatus.NOT_HOLDING:
            if set_marker_success:
                if self.prev_dist_to_goal is not None:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                else:
                    diff_dist = 0.0
                dist_reward = diff_dist * self._config.DIST_REWARD
                reward += dist_reward
            elif self.prev_success:
                pass
            else:
                if self.prev_dist_to_goal is not None:
                    diff_dist = self.prev_dist_to_goal - gripper_to_marker_dist
                    diff_dist = round(diff_dist, 3)
                else:
                    diff_dist = 0.0
                dist_reward = diff_dist * self._config.DIST_REWARD
                reward += dist_reward

        if set_marker_success:
            self.prev_dist_to_goal = gripper_to_resting_dist
            if not self.prev_success:
                reward += self._config.SUCC_REWARD
        else:
            self.prev_dist_to_goal = gripper_to_marker_dist
            if self.prev_success:
                task._is_episode_active = False
                # print("Interesting things happen:", marker_to_goal_dist)
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_INV_SUCC", False
                )
        self.prev_success = set_marker_success

        # set marker reward
        if self.prev_dist_to_goal2 is not None:
            diff_dist2 = self.prev_dist_to_goal2 - marker_to_goal_dist
            diff_dist2 = round(diff_dist2, 3)
        else:
            diff_dist2 = 0.0
        if "DIST_REWARD2" in self._config:
            reward += diff_dist2 * self._config.DIST_REWARD2
        else:
            reward += diff_dist2 * self._config.DIST_REWARD
        self.prev_dist_to_goal2 = marker_to_goal_dist

        self._metric = reward


@registry.register_measure
class RearrangeSetMarkerRewardV1(RearrangeSetMarkerReward):
    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures

        marker_to_goal_dist = measures[
            MarkerToGoalDistance.cls_uuid
        ].get_metric()
        gripper_to_marker_dist = measures[
            GripperToMarkerDistance.cls_uuid
        ].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()

        gs_measure = measures[GripperStatusMeasureV1.cls_uuid]
        gripper_status = gs_measure.status
        n_pick_correct = gs_measure.get_metric()["pick_correct"]
        n_drop = gs_measure.get_metric()["drop"]
        set_marker_success = measures[SetMarkerSuccess.cls_uuid].get_metric()
        # print("gripper_status", gripper_status)

        reward = 0.0

        if gripper_status == GripperStatus.PICK_CORRECT:
            if n_pick_correct == 1:  # first pick
                reward += self._config.PICK_REWARD

        if gripper_status == GripperStatus.PICK_WRONG:
            reward -= self._config.PICK_PENALTY
            if self._config.END_PICK_WRONG:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_PICK_WRONG", False
                )

        # Only allow open/close by using arm
        if gripper_status == GripperStatus.HOLDING_CORRECT:
            if self.prev_dist_to_goal2 is not None:
                diff_dist2 = self.prev_dist_to_goal2 - marker_to_goal_dist
                diff_dist2 = round(diff_dist2, 3)
            else:
                diff_dist2 = 0.0

            max_qvel = self._config.get("MAX_QVEL", -1.0)
            if max_qvel > 0 and diff_dist2 > max_qvel:
                # diff_dist2 = max_qvel
                task._is_episode_active = False

            if "DIST_REWARD2" in self._config:
                reward += diff_dist2 * self._config.DIST_REWARD2
            else:
                reward += diff_dist2 * self._config.DIST_REWARD
            self.prev_dist_to_goal2 = marker_to_goal_dist

        if gripper_status == GripperStatus.HOLDING_WRONG:
            raise RuntimeError

        if gripper_status == GripperStatus.DROP:
            if set_marker_success:
                if n_pick_correct > 0 and n_drop == 1:
                    reward += self._config.DROP_REWARD
            else:
                reward -= self._config.DROP_PENALTY
                if self._config.END_DROP:
                    task._is_episode_active = False
                    task._is_episode_truncated = self._config.get(
                        "TRUNCATE_DROP", False
                    )

        if gripper_status == GripperStatus.NOT_HOLDING:
            if set_marker_success:
                if self.prev_dist_to_goal is not None:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                else:
                    diff_dist = 0.0
                dist_reward = diff_dist * self._config.DIST_REWARD
                reward += dist_reward
            elif self.prev_success:
                pass
            else:
                if self.prev_dist_to_goal is not None:
                    diff_dist = self.prev_dist_to_goal - gripper_to_marker_dist
                    diff_dist = round(diff_dist, 3)
                else:
                    diff_dist = 0.0
                dist_reward = diff_dist * self._config.DIST_REWARD
                reward += dist_reward

        if set_marker_success:
            self.prev_dist_to_goal = gripper_to_resting_dist
            if not self.prev_success:
                reward += self._config.SUCC_REWARD
        else:
            self.prev_dist_to_goal = gripper_to_marker_dist
            if self.prev_success:
                task._is_episode_active = False
                # print("Interesting things happen:", marker_to_goal_dist)
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_INV_SUCC", False
                )
        self.prev_success = set_marker_success

        self._metric = reward
