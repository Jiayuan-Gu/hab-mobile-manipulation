from typing import Dict, List, Optional, Tuple, Union

import attr
import magnum as mn
import numpy as np
from habitat import Simulator
from habitat_sim.physics import (
    JointMotorSettings,
    ManagedBulletArticulatedObject,
)
from habitat_sim.sensor import Sensor

from habitat_extensions.utils import art_utils, mn_utils


@attr.s(auto_attribs=True, slots=True)
class RobotCameraParams:
    cam_pos: mn.Vector3
    look_at: mn.Vector3
    up: mn.Vector3 = mn.Vector3(0, 1, 0)  # +y is upward in habitat
    link_id: int = -1  # the link the camera is mounted


@attr.s(auto_attribs=True, slots=True)
class RobotParams:
    urdf_path: str

    arm_joints: List[int]
    gripper_joints: List[int]
    arm_init_params: Optional[List[float]]
    gripper_init_params: Optional[List[float]]

    ee_link: int
    ee_offset: mn.Vector3

    # joint_stiffness: float
    # joint_damping: float
    # joint_max_impulse: float
    arm_pos_gain: Union[List[float], float]
    arm_vel_gain: Union[List[float], float]
    arm_max_impulse: Union[List[float], float]

    cameras: Dict[str, RobotCameraParams]

    fixed_base: bool = True
    auto_clamp_joint_limits: bool = True
    extra: dict = attr.ib(factory=dict)


class Robot:
    sim_obj: ManagedBulletArticulatedObject

    # cache mapping for better efficiency
    # NOTE(jigu): joint_pos_offset are used for joint positions
    # NOTE(jigu): dof_offset are used for joint velocities
    link_joint_pos_offset: Dict[int, int]
    link_num_joint_pos: Dict[int, int]
    link_dof_offset: Dict[int, int]
    link_num_dofs: Dict[int, int]
    joint_motors: Dict[int, Tuple[int, JointMotorSettings]]

    cameras: Dict[str, Tuple[int, mn.Matrix4, Sensor]]

    def __init__(self, sim: Simulator, params: RobotParams) -> None:
        self.sim = sim
        self.params = params

        self.sim_obj = None
        self.cameras = {}

    def reconfigure(self):
        """Initialize the robot (load urdf, initialize motors)."""
        art_obj_mgr = self.sim.get_articulated_object_manager()

        # remove the old one if exists
        if self.sim_obj is not None:
            assert self.sim_obj.is_alive
            art_obj_mgr.remove_object_by_id(self.sim_obj.object_id)
            self.sim_obj = None

        self.sim_obj = art_obj_mgr.add_articulated_object_from_urdf(
            self.params.urdf_path, fixed_base=self.params.fixed_base
        )
        self.sim_obj.awake = True
        # NOTE(jigu): I do not use @auto_clamp_joint_limits due to legacy issues.
        # self.sim_obj.auto_clamp_joint_limits = True

        # DEBUG(jigu): check robot info
        # self.print_info()

        # Cache mapping
        (
            self.link_joint_pos_offset,
            self.link_num_joint_pos,
            self.link_dof_offset,
            self.link_num_dofs,
        ) = art_utils.get_link_joint_mapping(self.sim_obj)
        self.arm_joint_pos_offset = art_utils.get_joint_pos_offset(
            self.link_joint_pos_offset,
            self.link_num_joint_pos,
            self.params.arm_joints,
        )
        self.arm_joint_dof_offset = art_utils.get_joint_pos_offset(
            self.link_dof_offset,
            self.link_num_dofs,
            self.params.arm_joints,
        )
        self.gripper_joint_pos_offset = art_utils.get_joint_pos_offset(
            self.link_joint_pos_offset,
            self.link_num_joint_pos,
            self.params.gripper_joints,
        )

        # NOTE(jigu): follow p-viz-plan, otherwise the arm can penetrate the counter.
        # # remove default damping motors
        # art_utils.remove_existing_joint_motors(self.sim_obj)
        # # set damping motors for all joints
        # art_utils.create_all_motors(
        #     self.sim_obj,
        #     pos_gain=self.params.joint_stiffness,
        #     vel_gain=self.params.joint_damping,
        #     max_impulse=self.params.joint_max_impulse,
        # )

        #  Cache joint motors. No new motors should be added after it.
        self._sync_joint_motors()

        # Set arm joint motors
        n_joints = len(self.params.arm_joints)
        arm_pos_gain = np.broadcast_to(self.params.arm_pos_gain, n_joints)
        arm_vel_gain = np.broadcast_to(self.params.arm_vel_gain, n_joints)
        arm_max_impulse = np.broadcast_to(
            self.params.arm_max_impulse, n_joints
        )
        for i, link_id in enumerate(self.params.arm_joints):
            self._update_motor(
                link_id,
                position_gain=arm_pos_gain[i],
                velocity_gain=arm_vel_gain[i],
                max_impulse=arm_max_impulse[i],
            )

        # Set gripper joint motors (same param as arm)
        for i, link_id in enumerate(self.params.gripper_joints):
            self._update_motor(
                link_id,
                position_gain=arm_pos_gain[-1],
                velocity_gain=arm_vel_gain[-1],
                max_impulse=arm_max_impulse[-1],
            )

        # Initialize cameras
        self.cameras = {}
        for camera_prefix, camera_params in self.params.cameras.items():
            for uuid, sensor in self.sim._sensors.items():
                if uuid.startswith(camera_prefix):
                    cam2link = mn.Matrix4.look_at(
                        camera_params.cam_pos,
                        camera_params.look_at,
                        camera_params.up,
                    )
                    self.cameras[uuid] = (
                        camera_params.link_id,
                        cam2link,
                        sensor._sensor_object,
                    )

    def reset(self):
        """Reset the state of an existing robot."""
        self.sim_obj.clear_joint_states()

        self.reset_arm()

        # Reset motor target
        self._keep_joint_positions()

        self.sim_obj.awake = True
        self.sim_obj.clamp_joint_limits()

    def reset_arm(self):
        if self.params.arm_init_params is not None:
            self.arm_joint_pos = self.params.arm_init_params
            self.arm_motor_pos = self.params.arm_init_params
        if self.params.gripper_init_params is not None:
            self.gripper_joint_pos = self.params.gripper_init_params
            self.gripper_motor_pos = self.params.gripper_init_params

    def step(self):
        # NOTE(jigu): @awake should be set True every simulation step.
        self.sim_obj.awake = True
        self.sim_obj.clamp_joint_limits()

    def update_params(self, config: dict):
        params = self.params
        for k, v in config.items():
            setattr(params, k, v)
        self.params = params

    def get_state(self):
        state = {
            "T": self.sim_obj.transformation,
            "qpos": self.sim_obj.joint_positions,
            "qvel": self.sim_obj.joint_velocities,
            # Note that motor should be deeply copied!
            "motors": self._get_joint_motors(),
        }
        return state

    def set_state(self, state: dict):
        self.sim_obj.clear_joint_states()
        self.sim_obj.transformation = state["T"]
        self.sim_obj.joint_positions = state["qpos"]
        self.sim_obj.joint_velocities = state["qvel"]
        for joint_id, (motor_id, jms) in state["motors"].items():
            assert joint_id in self.joint_motors, joint_id
            self.sim_obj.update_joint_motor(motor_id, jms)
        self._sync_joint_motors()

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    def update_cameras(self):
        agent_inv_T = (
            self.sim._default_agent.scene_node.transformation.inverted()
        )
        for uuid, (_, _, sensor) in self.cameras.items():
            cam2world = self.get_cam_T(uuid)
            sensor.node.transformation = mn_utils.orthogonalize(
                agent_inv_T @ cam2world
            )

    def get_cam_T(self, uuid):
        link_id, cam2link, _ = self.cameras[uuid]
        if link_id != -1:
            link_T = self.sim_obj.get_link_scene_node(link_id).transformation
        else:
            # CAUTION(jigu): DO NOT use get_link_scene_node for base
            # link_T = self.sim_obj.transformation
            link_T = self.base_T
        return link_T @ cam2link

    # ---------------------------------------------------------------------------- #
    # Joint
    # ---------------------------------------------------------------------------- #
    def set_joint_pos(self, link_id, pos, update_motor=True):
        pos_offset = self.link_joint_pos_offset[link_id]
        assert self.link_num_joint_pos[link_id] == 1, link_id
        qpos = self.sim_obj.joint_positions
        qpos[pos_offset] = pos
        self.sim_obj.joint_positions = qpos

        if update_motor:
            self._update_motor(link_id, position_target=pos)

    # ---------------------------------------------------------------------------- #
    # Motor
    # ---------------------------------------------------------------------------- #
    def _get_joint_motors(self):
        joint_motors = {}
        for (
            motor_id,
            joint_id,
        ) in self.sim_obj.existing_joint_motor_ids.items():
            assert joint_id not in joint_motors, joint_id
            joint_motors[joint_id] = (
                motor_id,
                self.sim_obj.get_joint_motor_settings(motor_id),
            )
        return joint_motors

    def _sync_joint_motors(self):
        """Update the cache of JointMotorSettings."""
        self.joint_motors = self._get_joint_motors()

    def _update_motor(self, link_id, **kwargs):
        assert len(kwargs) > 0
        motor_id, jms = self.joint_motors[link_id]
        for k, v in kwargs.items():
            setattr(jms, k, v)
        self.sim_obj.update_joint_motor(motor_id, jms)
        return jms

    def _keep_joint_positions(self, joint_positions=None):
        if joint_positions is None:
            joint_positions = self.sim_obj.joint_positions
        for (
            motor_id,
            joint_id,
        ) in self.sim_obj.existing_joint_motor_ids.items():
            pos_offset = self.link_joint_pos_offset[joint_id]
            assert self.link_num_dofs[joint_id] == 1
            self._update_motor(
                joint_id,
                position_target=joint_positions[pos_offset],
                velocity_target=0.0,
            )

    # ---------------------------------------------------------------------------- #
    # Arm
    # ---------------------------------------------------------------------------- #
    @property
    def arm_joint_pos(self) -> List[float]:
        """Get the current arm joint positions."""
        joint_positions = self.sim_obj.joint_positions
        return [joint_positions[i] for i in self.arm_joint_pos_offset]

    @arm_joint_pos.setter
    def arm_joint_pos(self, joint_pos: List[float]):
        """Set the arm joint positions kinematically."""
        art_utils.set_joint_pos(
            self.sim_obj, self.arm_joint_pos_offset, joint_pos
        )

    @property
    def arm_joint_limits(self) -> Tuple[List[float], List[float]]:
        low, high = self.sim_obj.joint_position_limits
        low = [low[i] for i in self.arm_joint_pos_offset]
        high = [high[i] for i in self.arm_joint_pos_offset]
        return low, high

    @property
    def arm_joint_vel(self) -> List[float]:
        """Get the current arm joint velocities."""
        joint_velocities = self.sim_obj.joint_velocities
        return [joint_velocities[i] for i in self.arm_joint_dof_offset]

    @property
    def arm_joint_force(self):
        joint_forces = self.sim_obj.joint_forces
        return [joint_forces[i] for i in self.arm_joint_dof_offset]

    @arm_joint_force.setter
    def arm_joint_force(self, joint_forces: List[float]):
        art_utils.set_joint_force(
            self.sim_obj, self.arm_joint_dof_offset, joint_forces
        )

    @property
    def arm_motor_pos(self) -> List[float]:
        """Get the current position target of the arm joint motors."""
        position_targets = []
        for link_id in self.params.arm_joints:
            # assume each joint is 1-DoF
            position_targets.append(
                self.joint_motors[link_id][1].position_target
            )
        return position_targets

    @arm_motor_pos.setter
    def arm_motor_pos(self, position_targets: List[float]):
        """Set the desired position target of the arm joint motors."""
        assert len(position_targets) == len(self.params.arm_joints)
        for i, link_id in enumerate(self.params.arm_joints):
            self._update_motor(link_id, position_target=position_targets[i])

    @property
    def arm_motor_vel(self) -> List[float]:
        """Get the current velocity target of the arm joint motors."""
        velocity_targets = []
        for link_id in self.params.arm_joints:
            # assume each joint is 1-DoF
            velocity_targets.append(
                self.joint_motors[link_id][1].velocity_target
            )
        return velocity_targets

    @arm_motor_vel.setter
    def arm_motor_vel(self, velocity_targets: List[float]):
        """Set the desired velocity target of the arm joint motors."""
        assert len(velocity_targets) == len(self.params.arm_joints)
        for i, link_id in enumerate(self.params.arm_joints):
            self._update_motor(link_id, velocity_target=velocity_targets[i])

    # ---------------------------------------------------------------------------- #
    # Gripper
    # ---------------------------------------------------------------------------- #
    @property
    def gripper_joint_pos(self) -> List[float]:
        """Get the current gripper joint positions."""
        return [
            self.sim_obj.joint_positions[i]
            for i in self.gripper_joint_pos_offset
        ]

    @gripper_joint_pos.setter
    def gripper_joint_pos(self, joint_pos: List[float]):
        """Set the gripper joint positions kinematically."""
        art_utils.set_joint_pos(
            self.sim_obj, self.gripper_joint_pos_offset, joint_pos
        )

    @property
    def gripper_motor_pos(self) -> List[float]:
        """Get the current position target of the gripper joint motors."""
        position_targets = []
        for link_id in self.params.gripper_joints:
            # assume each joint is 1-DoF
            position_targets.append(
                self.joint_motors[link_id][1].position_target
            )
        return position_targets

    @gripper_motor_pos.setter
    def gripper_motor_pos(self, position_targets: List[float]):
        """Set the desired position target of the gripper joint motors."""
        assert len(position_targets) == len(self.params.gripper_joints)
        for i, link_id in enumerate(self.params.gripper_joints):
            self._update_motor(link_id, position_target=position_targets[i])

    # ---------------------------------------------------------------------------- #
    # Base
    # ---------------------------------------------------------------------------- #
    @property
    def base_T(self):
        return self.sim_obj.transformation

    @base_T.setter
    def base_T(self, T: mn.Matrix4):
        self.sim_obj.transformation = T

    @property
    def base_pos(self):
        return np.array(self.sim_obj.translation, dtype=np.float32)

    @base_pos.setter
    def base_pos(self, position: Union[List[float], np.ndarray]):
        self.sim_obj.translation = mn.Vector3(position)

    @property
    def base_ori(self) -> float:
        """The (y-axis) orientation of the base."""
        rotation = self.sim_obj.rotation
        heading_vector = rotation.transform_vector(mn.Vector3(1, 0, 0))
        return np.arctan2(-heading_vector[2], heading_vector[0])

    @base_ori.setter
    def base_ori(self, angle: float):
        self.sim_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(angle), mn.Vector3(0, 1, 0)
        )

    # ---------------------------------------------------------------------------- #
    # Transformation
    # ---------------------------------------------------------------------------- #
    @property
    def ee_T(self):
        return self.sim_obj.get_link_scene_node(
            self.params.ee_link
        ).transformation

    @property
    def gripper_T(self):
        gripper_T = self.ee_T
        gripper_T.translation = gripper_T.transform_point(
            self.params.ee_offset
        )
        return gripper_T

    @property
    def gripper_pos(self):
        return np.array(self.gripper_T.translation, dtype=np.float32)

    # ---------------------------------------------------------------------------- #
    # Others
    # ---------------------------------------------------------------------------- #
    @property
    def object_id(self) -> int:
        return self.sim_obj.object_id

    def print_info(self):
        print("*" * 8, "Links", "*" * 8)
        print(art_utils.get_links_info(self.sim_obj))
        print("*" * 8, "Joints", "*" * 8)
        print(art_utils.get_joints_info(self.sim_obj))
        print("*" * 8, "Motors", "*" * 8)
        print(art_utils.get_joint_motors_info(self.sim_obj))

    def set_semantic_ids(self, semantic_id):
        for node in self.sim_obj.visual_scene_nodes:
            node.semantic_id = semantic_id
