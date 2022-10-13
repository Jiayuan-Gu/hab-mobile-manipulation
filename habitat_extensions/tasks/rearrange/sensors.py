from enum import Enum, auto
from typing import Dict

import magnum as mn
import numpy as np
from gym import spaces
from habitat import logger
from habitat.config.default import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes

from .sim import RearrangeSim
from .task import RearrangeEpisode, RearrangeTask


class MySensor(Sensor):
    def __init__(self, *args, sim: RearrangeSim, config: Config, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        if hasattr(self, "cls_uuid"):
            return self.cls_uuid
        else:
            raise NotImplementedError

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR


class PositionSensor(MySensor):
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def _get_world_position(self, *args, **kwargs):
        """Return a position in the world frame."""
        raise NotImplementedError()

    def get_observation(self, *args, task: RearrangeTask, **kwargs):
        position = self._get_world_position(*args, task=task, **kwargs)
        position = mn.Vector3(position)

        robot = self._sim.robot
        frame = self.config.get("FRAME", "world")

        if frame == "world":
            T = mn.Matrix4.identity_init()
        elif frame == "base":
            T = robot.base_T.inverted()
        elif frame == "gripper":
            T = robot.gripper_T.inverted()
        elif frame == "base_t":
            T = mn.Matrix4.translation(-robot.base_T.translation)
        elif frame == "start_base":
            T = task.start_base_T.inverted()
        else:
            raise NotImplementedError(frame)

        position = T.transform_point(position)
        return np.array(position, dtype=np.float32)


class MyMeasure(Measure):
    def __init__(self, *args, sim: RearrangeSim, config: Config, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args, **kwargs) -> str:
        if "UUID" in self._config:
            return self._config["UUID"]
        if hasattr(self, "cls_uuid"):
            return self.cls_uuid
        else:
            raise NotImplementedError

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def _check_after_reset(
        self, *args, task: RearrangeTask, episode: RearrangeEpisode, **kwargs
    ):
        """Check episode status after reset (especially with termination)."""
        if not task.is_episode_active:
            logger.warning(
                "Episode {} is not active after {} when reset.".format(
                    episode.episode_id, self.uuid
                )
            )
        if task.is_episode_truncated:
            logger.warning(
                "Episode {} is truncated after {} when reset.".format(
                    episode.episode_id, self.uuid
                )
            )


# ---------------------------------------------------------------------------- #
# Sensor
# ---------------------------------------------------------------------------- #
@registry.register_sensor
class ArmJointPositionSensor(MySensor):
    cls_uuid = "arm_joint_pos"

    def _get_observation_space(self, *args, **kwargs):
        n_qpos = len(self._sim.robot.params.arm_init_params)
        return spaces.Box(
            shape=(n_qpos,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def get_observation(self, *args, **kwargs):
        return np.array(self._sim.robot.arm_joint_pos, dtype=np.float32)


@registry.register_sensor
class ArmJointVelocitySensor(MySensor):
    cls_uuid = "arm_joint_vel"

    def _get_observation_space(self, *args, **kwargs):
        n_qpos = len(self._sim.robot.params.arm_init_params)
        return spaces.Box(
            shape=(n_qpos,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def get_observation(self, *args, **kwargs):
        return np.array(self._sim.robot.arm_joint_vel, dtype=np.float32)


@registry.register_sensor
class IsGraspedSensor(MySensor):
    cls_uuid = "is_grasped"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, *args, **kwargs):
        return np.array(
            self._sim.gripper.is_grasped, dtype=np.float32
        ).reshape((1,))


@registry.register_sensor
class GripperPositionSensor(PositionSensor):
    cls_uuid = "gripper_pos"

    def _get_world_position(self, *args, **kwargs):
        # print(self._sim.robot.sim_obj.transformation.inverted() @ self._sim.robot.ee_transform)
        return self._sim.robot.gripper_pos


@registry.register_sensor
class PickGoalSensor(PositionSensor):
    cls_uuid = "pick_goal"

    def _get_world_position(self, *args, task: RearrangeTask, **kwargs):
        return task.pick_goal


@registry.register_sensor
class PlaceGoalSensor(PositionSensor):
    cls_uuid = "place_goal"

    def _get_world_position(self, *args, task: RearrangeTask, **kwargs):
        return task.place_goal


@registry.register_sensor
class RestingPositionSensor(MySensor):
    cls_uuid = "resting_pos"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def get_observation(self, *args, task: RearrangeTask, **kwargs):
        return task.resting_position  # base frame


# ---------------------------------------------------------------------------- #
# Measure
# ---------------------------------------------------------------------------- #
@registry.register_measure
class GripperToObjectDistance(MyMeasure):
    cls_uuid = "gripper_to_obj_dist"

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        gripper_pos = self._sim.robot.gripper_pos
        obj_pos = np.array(task.tgt_obj.translation, dtype=np.float32)
        dist = np.linalg.norm(obj_pos - gripper_pos)
        self._metric = dist

        # DEBUG(jigu): object suddenly move due to base movement
        # if dist > 5.0:
        #     print("*" * 10)
        #     episode = task._sim.habitat_config["EPISODE"]
        #     episode_id = episode["episode_id"]
        #     print("Episode", episode_id)
        #     print(dist, self._sim.robot.base_pos, obj_pos, task.pick_goal)
        #     print("*" * 10)


@registry.register_measure
class GripperToRestingDistance(MyMeasure):
    cls_uuid = "gripper_to_resting_dist"

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        gripper_pos = self._sim.robot.gripper_pos
        resting_pos = self._sim.robot.transform(
            task.resting_position, "base2world"
        )
        dist = np.linalg.norm(resting_pos - gripper_pos)
        self._metric = dist


@registry.register_measure
class ObjectToGoalDistance(MyMeasure):
    cls_uuid = "obj_to_goal_dist"

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        obj_pos = np.array(task.tgt_obj.translation, dtype=np.float32)
        goal_pos = task.place_goal
        dist = np.linalg.norm(goal_pos - obj_pos)
        self._metric = dist


class GripperStatus(Enum):
    NOT_HOLDING = auto()
    PICK_CORRECT = auto()
    PICK_WRONG = auto()
    HOLDING_CORRECT = auto()
    HOLDING_WRONG = auto()
    DROP = auto()


@registry.register_measure(name="GripperStatus")
class GripperStatusMeasure(MyMeasure):
    cls_uuid = "gripper_status"
    status: int
    counts: Dict[int, int]

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        # TODO(jigu): initial status might depend on task
        self.status = GripperStatus.NOT_HOLDING
        self.counts = {member: 0 for member in GripperStatus}
        self._update(*args, task=task, **kwargs)

    def _update(self, *args, task: RearrangeTask, **kwargs):
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
                # if self._sim.gripper.grasped_obj == task.tgt_obj:
                if self._sim.gripper.grasped_obj_id == task.tgt_obj.object_id:
                    self.status = GripperStatus.PICK_CORRECT
                else:
                    self.status = GripperStatus.PICK_WRONG
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

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        self._update(*args, task=task, **kwargs)
        self.counts[self.status] += 1

    def get_metric(self):
        return {k.name.lower(): v for k, v in self.counts.items()}


@registry.register_measure
class InvalidGrasp(MyMeasure):
    cls_uuid = "invalid_grasp"

    def reset_metric(self, *args, **kwargs):
        self._metric = 0

    def update_metric(self, *args, **kwargs):
        thresh = self._config.get("THRESHOLD", 0.09)
        if self._sim.gripper.is_invalid_grasp(thresh):
            self._metric += 1


@registry.register_measure
class InvalidGraspV1(MyMeasure):
    cls_uuid = "invalid_grasp"

    def reset_metric(self, *args, **kwargs):
        self._metric = 0

    def update_metric(self, *args, **kwargs):
        if self._sim.gripper.grasped_obj is not None:
            # thresh = 0.09
            thresh = 1.0
        elif self._sim.gripper.grasped_marker is not None:
            thresh = 0.2
        else:
            thresh = None
        if self._sim.gripper.is_invalid_grasp(thresh):
            self._metric += 1
            self._sim.gripper.desnap(False)


@registry.register_measure
class InvalidGraspPenalty(MyMeasure):
    cls_uuid = "invalid_grasp_penalty"

    def reset_metric(self, *args, **kwargs):
        super().reset_metric(*args, **kwargs)
        self._check_after_reset(*args, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        thresh = self._config.get("THRESHOLD", 0.09)
        if self._sim.gripper.is_invalid_grasp(thresh):
            self._metric = -self._config.PENALTY
            if self._config.END_EPISODE:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_EPISODE", False
                )
        else:
            self._metric = 0.0


@registry.register_measure
class InvalidGraspPenaltyV1(MyMeasure):
    cls_uuid = "invalid_grasp_penalty"

    def reset_metric(self, *args, **kwargs):
        super().reset_metric(*args, **kwargs)
        self._check_after_reset(*args, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        if self._sim.gripper.grasped_obj is not None:
            thresh = self._config.get("OBJ_THRESHOLD", 0.09)
        elif self._sim.gripper.grasped_marker is not None:
            thresh = self._config.get("ART_THRESHOLD", 0.2)
        else:
            thresh = None

        if self._sim.gripper.is_invalid_grasp(thresh):
            self._metric = -self._config.PENALTY
            if self._config.END_EPISODE:
                task._is_episode_active = False
                task._is_episode_truncated = self._config.get(
                    "TRUNCATE_EPISODE", False
                )
        else:
            self._metric = 0.0


@registry.register_measure
class RobotForce(MyMeasure):
    cls_uuid = "robot_force"

    def reset_metric(self, *args, **kwargs):
        self._accum_force = 0.0
        self._prev_force = 0.0
        self._delta_force = None
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        curr_force = self._sim.get_robot_collision(
            self._config.INCLUDE_OBJ_COLLISIONS
        )
        delta_force = curr_force - self._prev_force
        if delta_force > self._config.MIN_DELTA_FORCE:
            self._accum_force += delta_force
            self._delta_force = delta_force
        elif delta_force < 0.0:
            self._prev_force = curr_force
            self._delta_force = 0.0
        else:
            # ignore small variation
            self._delta_force = 0.0
        self._metric = self._accum_force


@registry.register_measure
class ForcePenalty(MyMeasure):
    cls_uuid = "force_penalty"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [RobotForce.cls_uuid]
        )
        self._metric = 0.0

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        robot_force = measures[RobotForce.cls_uuid]
        accum_force = robot_force._accum_force
        delta_force = robot_force._delta_force

        force_penalty = min(
            delta_force * self._config.FORCE_PENALTY,
            self._config.MAX_FORCE_PENALTY,
        )

        # end if exceed max accumulated force
        if (
            self._config.MAX_ACCUM_FORCE >= 0.0
            and accum_force > self._config.MAX_ACCUM_FORCE
        ):
            force_penalty += self._config.MAX_ACCUM_FORCE_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_EPISODE", False
            )

        self._metric = -force_penalty


@registry.register_measure
class CollisionPenalty(MyMeasure):
    cls_uuid = "collision_penalty"

    def reset_metric(self, *args, **kwargs):
        # Hardcode all links except hand and gripper
        self.link_ids = list(range(21))
        return super().reset_metric(*args, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        curr_force = self._sim.get_robot_collision(
            False, link_ids=self.link_ids
        )

        if curr_force >= self._config.MAX_FORCE:
            # print("curr_force", curr_force)
            self._metric = -self._config.PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_EPISODE", False
            )
        else:
            self._metric = 0.0


@registry.register_measure
class PlaceObjectSuccess(MyMeasure):
    cls_uuid = "place_obj_success"

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [ObjectToGoalDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        measures = task.measurements.measures
        obj_to_goal_dist = measures[ObjectToGoalDistance.cls_uuid].get_metric()
        self._metric = obj_to_goal_dist <= self._config.THRESHOLD


@registry.register_measure
class ActionPenalty(MyMeasure):
    cls_uuid = "action_penalty"

    def reset_metric(self, *args, **kwargs):
        self._metric = 0.0

    def update_metric(self, *args, action: dict, **kwargs) -> None:
        a = action["action_args"][self._config.SUB_ACTION]
        self._metric = -np.mean(np.abs(a)) * self._config.PENALTY
