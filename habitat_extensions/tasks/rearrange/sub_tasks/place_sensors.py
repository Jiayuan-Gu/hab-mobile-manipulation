from habitat.core.registry import registry

# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
from ..sensors import (
    GripperStatus,
    GripperStatusMeasure,
    GripperToRestingDistance,
    MyMeasure,
    ObjectToGoalDistance,
    PlaceObjectSuccess,
)
from ..task import RearrangeTask


@registry.register_measure
class RearrangePlaceSuccess(MyMeasure):
    cls_uuid = "rearrange_place_success"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GripperToRestingDistance.cls_uuid,
                PlaceObjectSuccess.cls_uuid,
            ],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()
        rest_success = (
            not self._sim.gripper.is_grasped and dist <= self._config.THRESHOLD
        )
        self._metric = rest_success and place_success


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangePlaceReward(MyMeasure):
    prev_dist_to_goal: float
    cls_uuid = "rearrange_place_reward"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjectToGoalDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasure.cls_uuid,
                    PlaceObjectSuccess.cls_uuid,
                ],
            )

        self.prev_dist_to_goal = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gripper_status = measures[GripperStatusMeasure.cls_uuid].status
        # print("gripper_status", gripper_status)

        reward = 0.0

        if gripper_status == GripperStatus.DROP:
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status in [
            GripperStatus.PICK_CORRECT,
            GripperStatus.HOLDING_CORRECT,
        ]:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - obj_to_goal_dist
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = -obj_to_goal_dist * self._config.DIST_REWARD
            reward += dist_reward
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.NOT_HOLDING:
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
        else:
            raise RuntimeError(gripper_status)

        self._metric = reward


@registry.register_measure
class RearrangePlaceRewardV1(RearrangePlaceReward):
    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gs_measure = measures[GripperStatusMeasure.cls_uuid]
        n_drop = gs_measure.get_metric()["drop"]
        gripper_status = gs_measure.status
        # print("gripper_status", gripper_status)
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()

        reward = 0.0

        if gripper_status == GripperStatus.DROP:
            if place_success:
                if n_drop == 1:  # first drop
                    reward += self._config.RELEASE_REWARD
            else:
                reward -= self._config.RELEASE_PENALTY
                if self._config.END_DROP:
                    task._is_episode_active = False
                    task._is_episode_truncated = self._config.get(
                        "TRUNCATE_DROP", False
                    )
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_CORRECT:
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.HOLDING_CORRECT:
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - obj_to_goal_dist
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = -obj_to_goal_dist * self._config.DIST_REWARD
            reward += dist_reward
            self.prev_dist_to_goal = obj_to_goal_dist
        elif gripper_status == GripperStatus.NOT_HOLDING:
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
        elif gripper_status == GripperStatus.PICK_WRONG:
            # Only for composite reward
            reward -= self._config.PICK_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_PICK_WRONG", False
            )
        else:
            raise RuntimeError(gripper_status)

        self._metric = reward


@registry.register_measure
class RearrangePlaceRewardV2(RearrangePlaceReward):
    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjectToGoalDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasure.cls_uuid,
                    PlaceObjectSuccess.cls_uuid,
                ],
            )

        self.prev_obj_to_goal_dist = None
        self.prev_gripper_to_resting_dist = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        gs_measure = measures[GripperStatusMeasure.cls_uuid]
        gripper_status = gs_measure.status
        place_success = measures[PlaceObjectSuccess.cls_uuid].get_metric()

        reward = 0.0

        if place_success:
            reward += 0.01

        if gripper_status == GripperStatus.DROP:
            if obj_to_goal_dist >= 0.3:
                reward -= self._config.RELEASE_PENALTY
                if self._config.END_DROP:
                    task._is_episode_active = False
                    task._is_episode_truncated = self._config.get(
                        "TRUNCATE_DROP", False
                    )
            self.prev_gripper_to_resting_dist = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_CORRECT:
            pass
        elif gripper_status == GripperStatus.HOLDING_CORRECT:
            pass
        elif gripper_status == GripperStatus.NOT_HOLDING:
            if self._config.USE_DIFF:
                if self.prev_gripper_to_resting_dist is None:
                    diff_dist = 0.0
                else:
                    diff_dist = (
                        self.prev_gripper_to_resting_dist
                        - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            self.prev_gripper_to_resting_dist = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_WRONG:
            # Only for composite reward
            reward -= self._config.PICK_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_PICK_WRONG", False
            )
        else:
            raise RuntimeError(gripper_status)

        # Distance between object and goal
        if self._config.USE_DIFF:
            if self.prev_obj_to_goal_dist is None:
                diff_dist = 0.0
            else:
                diff_dist = self.prev_obj_to_goal_dist - obj_to_goal_dist
                diff_dist = round(diff_dist, 3)
            dist_reward = diff_dist * self._config.DIST_REWARD
        else:
            dist_reward = -obj_to_goal_dist * self._config.DIST_REWARD
        reward += dist_reward
        self.prev_obj_to_goal_dist = obj_to_goal_dist

        self._metric = reward
