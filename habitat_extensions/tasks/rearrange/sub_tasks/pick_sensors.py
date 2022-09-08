import numpy as np
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry

from ..sensors import (
    GripperStatus,
    GripperStatusMeasure,
    GripperToObjectDistance,
    GripperToRestingDistance,
    MyMeasure,
)
from ..task import RearrangeTask


# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
@registry.register_measure
class ReachObjectSuccess(MyMeasure):
    cls_uuid = "reach_obj_success"

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GripperToObjectDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        measures = task.measurements.measures
        gripper_to_obj_dist = measures[
            GripperToObjectDistance.cls_uuid
        ].get_metric()
        self._metric = gripper_to_obj_dist <= self._config.THRESHOLD


@registry.register_measure
class RearrangePickSuccess(MyMeasure):
    cls_uuid = "rearrange_pick_success"

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GripperToRestingDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()

        correct_grasp = (
            self._sim.gripper.grasped_obj_id == task.tgt_obj.object_id
        )
        self._metric = correct_grasp and dist <= self._config.THRESHOLD


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangePickReward(MyMeasure):
    prev_dist_to_goal: float
    cls_uuid = "rearrange_pick_reward"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    GripperToObjectDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasure.cls_uuid,
                ],
            )

        self.prev_dist_to_goal = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        gripper_to_obj_dist = measures[
            GripperToObjectDistance.cls_uuid
        ].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gripper_status = measures[GripperStatusMeasure.cls_uuid].status
        # print("gripper_status", gripper_status)

        reward = 0.0

        if gripper_status == GripperStatus.PICK_CORRECT:
            reward += self._config.PICK_REWARD
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_WRONG:
            reward -= self._config.PICK_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_PICK_WRONG", False
            )
        elif gripper_status == GripperStatus.NOT_HOLDING:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - gripper_to_obj_dist
                    diff_dist = round(diff_dist, 3)

                    # Avoid knocking the object away
                    diff_thresh = self._config.get("DIFF_THRESH", -1.0)
                    if diff_thresh > 0 and np.abs(diff_dist) > diff_thresh:
                        diff_dist = 0.0
                        reward -= self._config.DIFF_PENALTY
                        task._is_episode_active = False
                        task._is_episode_truncated = False

                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = -gripper_to_obj_dist * self._config.DIST_REWARD
            reward += dist_reward
            self.prev_dist_to_goal = gripper_to_obj_dist
        elif gripper_status == GripperStatus.HOLDING_CORRECT:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.HOLDING_WRONG:
            raise RuntimeError()
            # pass
        elif gripper_status == GripperStatus.DROP:
            reward -= self._config.DROP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_DROP", False
            )

        self._metric = reward
