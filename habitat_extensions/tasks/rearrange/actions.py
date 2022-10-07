from collections import OrderedDict

import habitat_sim
import magnum as mn
import numpy as np
from gym import spaces
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Observations

from habitat_extensions.utils import coll_utils

from .sim import RearrangeSim
from .sim_v1 import RearrangeSimV1
from .task import RearrangeTask


class AtomicAction(SimulatorTaskAction):
    def _step(self, *args, **kwargs) -> Observations:
        """Step without simulation."""
        raise NotImplementedError

    def step(self, *args, **kwargs):
        self._step(*args, **kwargs)
        return self._sim.step(None)


@registry.register_task_action
class EmptyAction(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.step(None)


@registry.register_task_action
class DummyAction(SimulatorTaskAction):
    @property
    def action_space(self):
        shape = self._config.SHAPE
        return spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)


# ---------------------------------------------------------------------------- #
# Arm
# ---------------------------------------------------------------------------- #
@registry.register_task_action
class ArmVelAction(AtomicAction):
    _sim: RearrangeSim

    @property
    def action_space(self):
        n_qpos = len(self._sim.robot.params.arm_init_params)
        return spaces.Box(shape=(n_qpos,), low=-1, high=1, dtype=np.float32)

    def _step(self, velocity_targets, *args, **kwargs):
        # TODO(jigu): add clip option in config
        velocity_targets = np.array(velocity_targets, dtype=np.float32)
        velocity_targets = np.clip(velocity_targets, -1.0, 1.0)
        self._sim.robot.arm_motor_vel = velocity_targets

    def get_action_args(self, action: np.ndarray):
        return {"velocity_targets": action}


@registry.register_task_action
class MagicGraspAction(AtomicAction):
    """Magic grasp."""

    _sim: RearrangeSim

    @property
    def action_space(self):
        return spaces.Box(shape=(1,), low=-1, high=1, dtype=np.float32)

    def _get_obj_to_grasp(self):
        gripper_pos = self._sim.robot.gripper_pos
        objs_pos = self._sim.get_rigid_objs_pos()
        assert len(objs_pos) > 0
        gripper_to_objs_dist = np.linalg.norm(objs_pos - gripper_pos, axis=-1)
        closest_idx = np.argmin(gripper_to_objs_dist)
        closest_dist = gripper_to_objs_dist[closest_idx]
        if closest_dist <= self._config.THRESHOLD:
            obj_to_grasp = self._sim.get_rigid_obj(closest_idx)
        else:
            obj_to_grasp = None
        return obj_to_grasp

    def _get_marker_to_grasp(self):
        gripper_pos = self._sim.robot.gripper_pos
        markers_pos = [m.pos for m in self._sim.markers.values()]
        assert len(markers_pos) > 0
        gripper_to_markers_dist = np.linalg.norm(
            markers_pos - gripper_pos, axis=-1
        )
        closest_idx = np.argmin(gripper_to_markers_dist)
        closest_dist = gripper_to_markers_dist[closest_idx]
        if closest_dist <= self._config.THRESHOLD:
            marker = self._sim.get_marker(closest_idx)
        else:
            marker = None
        return marker

    def _grasp(self):
        # Follow p-viz-plan to first grasp marker if possible
        marker_to_grasp = self._get_marker_to_grasp()
        if marker_to_grasp is not None:
            self._sim.gripper.snap_to_marker(marker_to_grasp)
            return

        obj_to_grasp = self._get_obj_to_grasp()
        if obj_to_grasp is not None:
            self._sim.gripper.snap_to_obj(obj_to_grasp)
            return

    def _step(self, gripper_action, *args, **kwargs):
        if gripper_action > 0.0:
            self._sim.robot.open_gripper()
        elif gripper_action < 0.0:
            self._sim.robot.close_gripper()
        is_grasped = self._sim.gripper.is_grasped
        if (
            gripper_action > 0.0
            and not is_grasped
            and (not self._config.get("DISABLE_GRASP", False))
        ):
            self._grasp()
        elif (
            gripper_action < 0.0
            and is_grasped
            and (not self._config.get("DISABLE_RELEASE", False))
        ):
            self._sim.gripper.desnap()


@registry.register_task_action
class SuctionGraspAction(AtomicAction):
    _sim: RearrangeSimV1

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._scene_obj_ids = [
            x.object_id for x in self._sim.rigid_objs.values()
        ]

    @property
    def action_space(self):
        return spaces.Box(shape=(1,), low=-1, high=1, dtype=np.float32)

    def _grasp(self):
        robot_id = self._sim.robot.object_id
        contact_points = self._sim.get_physics_contact_points()
        gripper_link_ids = self._sim.robot.params.gripper_joints

        # NOTE(jigu): I suspect there might be fake contacts (force = 0)
        contact_infos = coll_utils.get_contact_infos(
            contact_points, robot_id, link_ids=gripper_link_ids
        )
        if len(contact_infos) == 0:
            return

        # First, check rigid objects
        for obj_id in self._scene_obj_ids:
            for c in contact_infos:
                if c["object_id"] == obj_id:
                    ee_T = self._sim.robot.ee_T
                    rom = self._sim.get_rigid_object_manager()
                    obj = rom.get_object_by_id(obj_id)
                    rel_pos = ee_T.inverted().transform_point(obj.translation)
                    self._sim.gripper.snap_to_obj(
                        obj_id,
                        force=False,
                        rel_pos=rel_pos,
                        should_open_gripper=False,
                    )
                    return

        # Next, check markers
        markers = self._sim.markers
        for marker_uuid, marker in markers.items():
            for c in contact_infos:
                if (
                    c["object_id"] == marker.art_obj.object_id
                    and c["object_link_id"] == marker.link_id
                ):
                    self._sim.gripper.snap_to_marker(marker_uuid)
                    return

    def _step(self, gripper_action, *args, **kwargs):
        is_grasped = self._sim.gripper.is_grasped
        if (
            gripper_action > 0.0
            and not is_grasped
            and (not self._config.get("DISABLE_GRASP", False))
        ):
            self._grasp()
        elif (
            gripper_action < 0.0
            and is_grasped
            and (not self._config.get("DISABLE_RELEASE", False))
        ):
            self._sim.gripper.desnap()


@registry.register_task_action
class ArmGripperAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs) -> None:
        super().__init__(config=config, sim=sim, **kwargs)
        arm_action_init = registry.get_task_action(config.ARM_ACTION.TYPE)
        self.arm_action = arm_action_init(config=config.ARM_ACTION, sim=sim)
        gripper_action_init = registry.get_task_action(
            config.GRIPPER_ACTION.TYPE
        )
        self.gripper_action = gripper_action_init(
            config=config.GRIPPER_ACTION, sim=sim
        )

    def reset(self, *args, **kwargs) -> None:
        self.arm_action.reset(*args, **kwargs)
        self.gripper_action.reset(*args, **kwargs)

    @property
    def action_space(self):
        return spaces.Dict(
            OrderedDict(
                arm_action=self.arm_action.action_space,
                gripper_action=self.gripper_action.action_space,
            )
        )

    def step(self, arm_action, gripper_action, **kwargs):
        self.gripper_action._step(gripper_action, **kwargs)
        self.arm_action._step(arm_action, **kwargs)
        return self._sim.step(None)


@registry.register_task_action
class ArmEEAction(AtomicAction):
    """IK-based arm action.
    The control signal is the change of the end-effector position.
    The hyperparameters are from Habitat 2.0.

    Notes(jigu):
        It is better to add a new action,
        if you would like to change the robot or other hyperparams.
    """

    _sim: RearrangeSim

    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)

        # NOTE(jigu): pybullet uses urdf frame while habitat uses inertia frame.
        # The constraints are defined in pybullet space (z is up).
        self.ee_constraints = np.array([[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]])

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1.0, high=1.0, dtype=np.float32)

    def reset(self, *args, task: RearrangeTask, **kwargs):
        super().reset(*args, **kwargs)
        # self.ee_tgt_pos = np.array([0.5, 0.0, 1.0])
        self.ee_tgt_pos = task.start_ee_pos
        # self._sim.sync_pyb_robot()
        # self.ee_tgt_pos = self._sim.pyb_robot.ee_state[4]

    def _step(self, ee_rel_pos, **kwargs):
        ee_rel_pos = np.clip(ee_rel_pos, -1.0, 1.0)
        ee_rel_pos *= self._config.CTRL_SCALE

        self._sim.sync_pyb_robot()
        ee_tgt_pos = self.ee_tgt_pos + ee_rel_pos
        # clip to workspace
        ee_tgt_pos = np.clip(
            ee_tgt_pos,
            self.ee_constraints[:, 0],
            self.ee_constraints[:, 1],
        )
        # NOTE(jigu): IK iter is 20 by default, which is only enough for small motion
        arm_tgt_qpos = self._sim.pyb_robot.IK(ee_tgt_pos)
        # err = self._sim.pyb_robot.compute_IK_error(ee_tgt_pos, arm_tgt_qpos)

        self._sim.robot.arm_motor_pos = arm_tgt_qpos
        self.ee_tgt_pos = ee_tgt_pos

    def get_action_args(self, action: np.ndarray):
        return {"ee_rel_pos": action}


@registry.register_task_action
class ArmRelPosAction(AtomicAction):
    _sim: RearrangeSim

    # def reset(self, *args, **kwargs):
    #     low, high = self._sim.robot.arm_joint_limits
    #     low = np.where(np.isinf(low), -np.pi, low)
    #     high = np.where(np.isinf(high), np.pi, high)
    #     self.qlimit = high - low

    @property
    def action_space(self):
        n_qpos = len(self._sim.robot.params.arm_init_params)
        return spaces.Box(shape=(n_qpos,), low=-1, high=1, dtype=np.float32)

    def _step(self, delta_qpos, *args, **kwargs):
        delta_qpos = np.clip(delta_qpos, -1, 1) * self._config.CTRL_SCALE
        tgt_qpos = self._sim.robot.arm_motor_pos + delta_qpos
        self._sim.robot.arm_motor_pos = tgt_qpos

    def get_action_args(self, action: np.ndarray):
        return {"delta_qpos": action}


# ---------------------------------------------------------------------------- #
# Base
# ---------------------------------------------------------------------------- #
@registry.register_task_action
class BaseVelAction(AtomicAction):
    _sim: RearrangeSim

    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)

        self._robot = self._sim.robot

        self.vel_ctrl = habitat_sim.physics.VelocityControl()
        self.vel_ctrl.controlling_lin_vel = True
        self.vel_ctrl.lin_vel_is_local = True
        self.vel_ctrl.controlling_ang_vel = True
        self.vel_ctrl.ang_vel_is_local = True

    @property
    def action_space(self):
        return spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)

    def preprocess_velocity(self, velocity):
        lin_vel, ang_vel = velocity

        # normalize
        if not self._config.get("NORMALIZED", True):
            lin_vel = lin_vel / self._config.LIN_SCALE
            ang_vel = ang_vel / self._config.ANG_SCALE

        # clip to [-1, 1]
        lin_vel = np.clip(lin_vel, -1, 1)
        ang_vel = np.clip(ang_vel, -1, 1)

        if self._config.get("DISABLE_BACKWARD", False):
            lin_vel = lin_vel * 0.5 + 0.5

        lin_vel = lin_vel * self._config.LIN_SCALE
        ang_vel = ang_vel * self._config.ANG_SCALE
        return lin_vel, ang_vel

    def _step(self, velocity, *args, **kwargs):
        """Move the robot base according to navmesh.

        NOTE(jigu): In p-viz-plan, there are also:
            - Stop when velocity is small
            - Revert robot pose if it collides with other objects.
        """
        lin_vel, ang_vel = self.preprocess_velocity(velocity)

        # x-axis is forward and y-axis is up.
        self.vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        # Compute current and target base transformation
        base_T = self._robot.base_T
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(base_T.rotation()), base_T.translation
        )
        target_rigid_state = self.vel_ctrl.integrate_transform(
            self._sim.timestep, rigid_state
        )

        # Computes a valid navigable end point given a target translation on the NavMesh.
        # Uses the configured sliding flag
        target_position = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        # Update the robot base
        target_T = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), target_position
        )
        self._robot.base_T = target_T

        # Update the grasped object
        grasped_obj = self._sim.gripper.grasped_obj
        if self._config.get("UPDATE_GRASP", False) and grasped_obj is not None:
            grapsed_obj_T = grasped_obj.transformation
            rel_T = target_T @ base_T.inverted()
            grasped_obj.transformation = rel_T @ grapsed_obj_T

    def get_action_args(self, action: np.ndarray):
        return {"velocity": action}


@registry.register_task_action
class BaseDiscVelAction(BaseVelAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.possible_velocities = np.array(
            [
                [lin_vel, ang_vel]
                for lin_vel in np.linspace(-0.5, 1.0, 4)
                for ang_vel in np.linspace(-1.0, 1.0, 5)
            ]
        )

    @property
    def action_space(self):
        return spaces.Discrete(4 * 5)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.is_stop_called = False

    def _step(self, action: int, *args, task: RearrangeTask, **kwargs):
        assert isinstance(action, int), action
        velocity = self.possible_velocities[action]
        # print("velocity", velocity, action)

        if np.allclose(velocity, 0):
            self.is_stop_called = True
        else:
            self.is_stop_called = False

        if self._config.get("END_ON_STOP", True) and self.is_stop_called:
            task._should_terminate = True

        super()._step(velocity, *args, task=task, **kwargs)

    def get_action_args(self, action: np.ndarray):
        return {"action": action.item()}


@registry.register_task_action
class BaseVelStopAction(BaseVelAction):
    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def _step(self, velocity, stop, *args, task: RearrangeTask, **kwargs):
        self.is_stop_called = stop > 0
        if self._config.get("END_ON_STOP", True) and self.is_stop_called:
            task._should_terminate = True
        super()._step(velocity, *args, task=task, **kwargs)

    def get_action_args(self, action: np.ndarray):
        return {"velocity": action[:-1], "stop": action[-1].item()}


# -------------------------------------------------------------------------- #
# Base + Arm + Gripper
# -------------------------------------------------------------------------- #
@registry.register_task_action
class BaseArmGripperAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs) -> None:
        super().__init__(config=config, sim=sim, **kwargs)
        base_action_init = registry.get_task_action(config.BASE_ACTION.TYPE)
        self.base_action = base_action_init(config=config.BASE_ACTION, sim=sim)
        arm_action_init = registry.get_task_action(config.ARM_ACTION.TYPE)
        self.arm_action = arm_action_init(config=config.ARM_ACTION, sim=sim)
        gripper_action_init = registry.get_task_action(
            config.GRIPPER_ACTION.TYPE
        )
        self.gripper_action = gripper_action_init(
            config=config.GRIPPER_ACTION, sim=sim
        )

    def reset(self, *args, **kwargs) -> None:
        self.base_action.reset(*args, **kwargs)
        self.arm_action.reset(*args, **kwargs)
        self.gripper_action.reset(*args, **kwargs)

    @property
    def action_space(self):
        return spaces.Dict(
            OrderedDict(
                base_action=self.base_action.action_space,
                arm_action=self.arm_action.action_space,
                gripper_action=self.gripper_action.action_space,
            )
        )

    def step(self, base_action, arm_action, gripper_action, **kwargs):
        # The order might matter!
        self.gripper_action._step(gripper_action, **kwargs)
        self.arm_action._step(arm_action, **kwargs)
        self.base_action._step(base_action, **kwargs)
        return self._sim.step(None)
