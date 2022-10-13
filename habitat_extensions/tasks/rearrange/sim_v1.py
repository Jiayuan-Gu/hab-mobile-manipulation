#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import magnum as mn
import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.robots.fetch_suction import FetchSuctionRobot
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat_sim.physics import (
    JointMotorSettings,
    ManagedBulletArticulatedObject,
    ManagedBulletRigidObject,
    MotionType,
)
from habitat_sim.sim import SimulatorBackend

from habitat_extensions.robots.marker import Marker
from habitat_extensions.robots.pybullet_utils import PybulletRobot
from habitat_extensions.utils import art_utils, coll_utils, mn_utils, obj_utils
from habitat_extensions.utils.geo_utils import (
    invert_transformation,
    transform_points,
)
from habitat_extensions.utils.sim_utils import (
    get_navmesh_settings,
    get_object_handle_by_id,
)


class MyFetchSuctionRobot(FetchSuctionRobot):
    hab2pyb = mn.Matrix4(
        np.float32(
            [
                [1, 0, 0, -0.0036],
                [0, 0.0107961, -0.9999417, 0],
                [0, 0.9999417, 0.0107961, 0.0014],
                [0, 0, 0, 1],
            ],
        )
    )

    def reset_arm(self):
        if self.params.arm_init_params is not None:
            self.arm_joint_pos = self.params.arm_init_params
            self.arm_motor_pos = self.params.arm_init_params
        if self.params.gripper_init_params is not None:
            self.gripper_joint_pos = self.params.gripper_init_params
            self.gripper_motor_pos = self.params.gripper_init_params

    @property
    def object_id(self) -> int:
        return self.sim_obj.object_id

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
        # gripper_T = self.ee_T
        # gripper_T.translation = gripper_T.transform_point(
        #     self.params.ee_offset
        # )
        # return gripper_T
        return self.ee_T

    @property
    def gripper_pos(self):
        return np.array(self.gripper_T.translation, dtype=np.float32)

    def transform(self, x: np.ndarray, T: Union[str, np.ndarray]):
        """Transform point(s) (from habitat world frame) to specified frame."""
        if isinstance(T, str):
            if T == "world2base":
                T = np.array(self.base_T.inverted())
            elif T == "world2pbase":
                T = np.array(self.hab2pyb @ self.base_T.inverted())
            elif T == "base2pbase":
                T = np.array(self.hab2pyb)
            elif T == "base2world":
                T = np.array(self.base_T)
            elif T == "pbase2base":
                T = invert_transformation(self.hab2pyb)
            else:
                raise NotImplementedError(T)

        assert T.shape == (4, 4)
        return transform_points(x, T)

    # -------------------------------------------------------------------------- #
    # Get/set state
    # -------------------------------------------------------------------------- #
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


class MyRearrangeGraspManager(RearrangeGraspManager):
    @property
    def grasped_obj(self):
        if self._snapped_obj_id is None:
            self._snapped_obj = None
            return None
        if self._snapped_obj is None:
            self._snapped_obj = self.snap_rigid_obj
        return self._snapped_obj

    @property
    def grasped_obj_id(self):
        return self._snapped_obj_id

    @property
    def grasped_marker(self):
        if self._snapped_marker_id is None:
            return None
        return self._sim.markers[self._snapped_marker_id]

    @property
    def grasped_marker_id(self):
        return self._snapped_marker_id

    def is_invalid_grasp(self, thresh=None):
        return self.is_violating_hold_constraint()

    def update_object_to_grasp(self) -> None:
        self.grasped_obj.transformation = self._managed_robot.ee_T


@registry.register_simulator(name="RearrangeSim-v1")
class RearrangeSimV1(HabitatSim):
    # RIGID_OBJECT_DIR = "data/objects/ycb/configs"
    RIGID_OBJECT_DIR = "data/objects/ycb_1.1"
    PRIMITIVE_DIR = "habitat_extensions/assets/objects/primitives"

    def __init__(self, config: Config):
        super().__init__(config)

        # NOTE(jigu): The first episode is used to initialized the simulator
        # When `habitat.Env` is initialized.
        # NOTE(jigu): DO NOT set `_current_scene` to None.
        self._prev_scene_id = None
        self._prev_scene_dataset = config.SCENE_DATASET
        self._initial_state = None

        self._initialize_templates()
        self.navmesh_settings = get_navmesh_settings(self._get_agent_config())

        # objects
        self.rigid_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()
        self.art_objs: Dict[
            str, ManagedBulletArticulatedObject
        ] = OrderedDict()
        self.viz_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()
        self.markers: Dict[str, Marker] = OrderedDict()

        # robot
        self.robot = MyFetchSuctionRobot(
            self.habitat_config.AGENT_0.ROBOT_URDF, self
        )
        ARM_URDF = "habitat_extensions/assets/robots/hab_fetch/robots/hab_fetch_arm_v2.urdf"
        self.pyb_robot = PybulletRobot(
            ARM_URDF, joint_indices=[0, 1, 2, 3, 4, 5, 6], ee_link_idx=8
        )
        self.gripper = MyRearrangeGraspManager(
            self, self.habitat_config, self.robot
        )

    def _initialize_templates(self):
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(self.RIGID_OBJECT_DIR)
        # primitives for visualization
        obj_attr_mgr.load_configs(self.PRIMITIVE_DIR)
        # print(obj_attr_mgr.get_template_handles())

    @property
    def timestep(self):
        return self.habitat_config.CONTROL_FREQ / self.habitat_config.SIM_FREQ

    @property
    def verbose(self):
        return self.habitat_config.get("VERBOSE", False)

    # -------------------------------------------------------------------------- #
    # Interfaces
    # -------------------------------------------------------------------------- #
    def reconfigure(self, habitat_config: Config):
        """Called before sim.reset() in `habitat.Env`."""
        # NOTE(jigu): release before super().reconfigure()
        # otherwise, there might be memory leak for constraint.
        # This extra release might also change results, but the reason is unknown.
        self.gripper.desnap(True)

        # NOTE(jigu): DO NOT use self._current_scene to judge
        is_same_scene = habitat_config.SCENE == self._prev_scene_id
        if self.verbose:
            print("is_same_scene", is_same_scene)

        is_same_scene_dataset = (
            habitat_config.SCENE_DATASET == self._prev_scene_dataset
        )

        # The simulator backend will be reconfigured.
        # Assets are invalid after a new scene is configured.
        # Note that ReplicaCAD articulated objects are managed by the backend.
        super().reconfigure(habitat_config)
        self._prev_scene_id = habitat_config.SCENE
        self._prev_scene_dataset = habitat_config.SCENE_DATASET

        # Acquire GL context for async rendering
        # if self.habitat_config.CONCUR_RENDER:
        #     self.renderer.acquire_gl_context()

        if not is_same_scene:
            self.art_objs = OrderedDict()
            self.rigid_objs = OrderedDict()
            self.robot.sim_obj = None
            self._initial_state = None

        if not is_same_scene_dataset:
            self._initialize_templates()

        # Called before new assets are added
        self.gripper.reconfigure()
        if not is_same_scene:
            self.robot.reconfigure()
            # print(art_utils.get_links_info(self.robot.sim_obj))
            # self.robot.set_semantic_ids(100)
        # elif self._initial_state is not None:
        #     self.robot.set_state(self._initial_state["robot_state"])

        if not is_same_scene:
            self._add_articulated_objects()
            self._initialize_articulated_objects()
        elif self._initial_state is not None:
            art_objs_qpos = self._initial_state["art_objs_qpos"]
            for handle, qpos in art_objs_qpos.items():
                art_obj = self.art_objs[handle]
                art_obj.clear_joint_states()  # joint positions are also zeroed.
                art_obj.joint_positions = qpos

        self._remove_rigid_objects()
        self._add_rigid_objects()
        self._add_markers()
        self._add_targets()

        assert len(self.viz_objs) == 0, self.viz_objs
        self.viz_objs = OrderedDict()

        if self.habitat_config.get("AUTO_SLEEP", False):
            self.sleep_all_objects()

        if not is_same_scene:
            # TODO(jigu): check whether navmesh can be auto loaded.
            # self._recompute_navmesh()
            self._cache_largest_island()

        # # Cache initial state
        # self._initial_state = self.get_state()

    def _add_rigid_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        episode = self.habitat_config.EPISODE

        for obj_info in episode["rigid_objs"]:
            template_handle = osp.join(self.RIGID_OBJECT_DIR, obj_info[0])
            obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
            T = mn.Matrix4(np.array(obj_info[1]))
            # obj.transformation = T
            obj.transformation = mn_utils.orthogonalize(T)
            obj.motion_type = MotionType.DYNAMIC
            self.rigid_objs[obj.handle] = obj
            if self.verbose:
                print("Add a rigid body", obj.handle, obj.object_id)

    def _remove_rigid_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle, obj in self.rigid_objs.items():
            assert obj.is_alive, handle
            if self.verbose:
                print(
                    "Remove a rigid object",
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        self.rigid_objs = OrderedDict()

    def _add_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            if handle == self.robot.sim_obj.handle:  # ignore robot
                continue
            self.art_objs[handle] = art_obj_mgr.get_object_by_handle(handle)

    def _remove_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for art_obj in self.art_objs.values():
            assert art_obj.is_alive
            if self.verbose:
                print(
                    "Remove an articulated object",
                    art_obj.handle,
                    art_obj.object_id,
                    art_obj.is_alive,
                )
            art_obj_mgr.remove_object_by_id(art_obj.object_id)
        self.art_objs = OrderedDict()

    def _initialize_articulated_objects(self):
        # NOTE(jigu): params from p-viz-plan/orp/sim.py
        for handle in self.art_objs:
            art_obj = self.art_objs[handle]
            for motor_id, link_id in art_obj.existing_joint_motor_ids.items():
                art_utils.update_motor(
                    art_obj, motor_id, velocity_gain=0.3, max_impulse=1.0
                )

    def _set_articulated_objects_from_episode(self):
        episode = self.habitat_config.EPISODE
        art_obj_mgr = self.get_articulated_object_manager()

        for handle, joint_states in episode["ao_states"].items():
            # print(handle, joint_states)
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            qpos = art_obj.joint_positions
            for link_id, joint_state in joint_states.items():
                pos_offset = art_obj.get_link_joint_pos_offset(int(link_id))
                qpos[pos_offset] = joint_state
            art_obj.joint_positions = qpos

    def print_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            print(handle, art_obj, art_obj.object_id)

    def _add_markers(self):
        self.markers = OrderedDict()
        art_obj_mgr = self.get_articulated_object_manager()

        # NOTE(jigu): The official one does not include all markers
        # episode = self.habitat_config.EPISODE
        # for marker_info in episode["markers"]:
        #     name = marker_info["name"]
        #     params = marker_info["params"]
        #     art_obj = art_obj_mgr.get_object_by_handle(params["object"])
        #     link_id = art_utils.get_link_id_by_name(art_obj, params["link"])
        #     marker = Marker(name, art_obj, link_id, params["offset"])
        #     self.markers[name] = marker

        fridge_handle = "fridge_:0000"
        art_obj = art_obj_mgr.get_object_by_handle(fridge_handle)
        link_id = art_utils.get_link_id_by_name(art_obj, "top_door")
        marker = Marker(
            "fridge_push_point", art_obj, link_id, offset=[0.10, -0.62, 0.2]
        )
        self.markers[marker.uuid] = marker

        drawer_handle = "kitchen_counter_:0000"
        art_obj = art_obj_mgr.get_object_by_handle(drawer_handle)
        drawer_link_names = [
            "drawer1_bottom",
            "drawer1_top",
            "drawer2_bottom",
            "drawer2_middle",
            "drawer2_top",
            "drawer4",
            "drawer3",
        ]
        for idx, link_name in enumerate(drawer_link_names):
            link_id = art_utils.get_link_id_by_name(art_obj, link_name)
            marker_name = "cab_push_point_{}".format(idx + 1)
            marker = Marker(marker_name, art_obj, link_id, offset=[0.3, 0, 0])
            self.markers[marker.uuid] = marker

    def _add_targets(self):
        self.targets = OrderedDict()
        episode = self.habitat_config.EPISODE
        # handles = sorted(episode["targets"].keys())
        # NOTE(jigu): The order of targets is used in `target_receptacles` and `goal_receptacles`
        handles = list(episode["targets"].keys())
        for handle in handles:
            T = episode["targets"][handle]
            self.targets[handle] = mn_utils.orthogonalize(T)

    def _recompute_navmesh(self):
        # Set all articulated objects static
        motion_types = OrderedDict()
        for handle, art_obj in self.art_objs.items():
            motion_types[handle] = art_obj.motion_type
            art_obj.motion_type = MotionType.STATIC

        # Recompute navmesh
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )

        # Restore motion type
        for handle, motion_type in motion_types.items():
            self.art_objs[handle].motion_type = motion_type

        # self.pathfinder.save_nav_mesh(navmesh_path)

        self._cache_largest_island()

    def _cache_largest_island(self):
        navmesh_vertices = np.stack(
            self.pathfinder.build_navmesh_vertices(), axis=0
        )
        self._largest_island_radius = max(
            [self.pathfinder.island_radius(p) for p in navmesh_vertices]
        )

    def is_at_larget_island(self, position, eps=1e-4):
        assert self.pathfinder.is_navigable(position), position
        island_raidus = self.pathfinder.island_radius(position)
        return np.abs(island_raidus - self._largest_island_radius) <= eps

    def sleep_all_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle in rigid_obj_mgr.get_object_handles():
            obj = rigid_obj_mgr.get_object_by_handle(handle)
            obj.awake = False

        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            art_obj.awake = False

    def reset(self):
        # # The agent and sensors are reset.
        # super().reset()

        # Minimal reset
        SimulatorBackend.reset(self)
        for i in range(len(self.agents)):
            self.reset_agent(i)

        # Uncomment if the simulator is reset but not reconfigured
        # self.set_state(self._initial_state)

        # Reset the articulated objects
        self._set_articulated_objects_from_episode()

        # Reset the robot
        self.robot.reset()

        # Place the robot
        # NOTE(jigu): I will set `start_position` out of the room,
        # so that some articulated objects can be initialized in tasks.
        episode = self.habitat_config.EPISODE
        self.robot.base_T = mn_utils.to_Matrix4(
            episode["start_position"], episode["start_rotation"]
        )

        # Reset the gripper
        self.gripper.reset()

        # Sync before getting observations
        # self.sync_agent()
        self.sync_pyb_robot()

        return None

    def get_observations(self):
        self.gripper.update()
        # self.robot.update_cameras()
        self.robot.update()
        self._prev_sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def internal_step(self, dt=None):
        """Internal simulation step."""
        if dt is None:
            dt = 1.0 / self.habitat_config.SIM_FREQ
        self.step_world(dt)
        # self.robot.step()

    def internal_step_by_time(self, seconds):
        steps = int(seconds * self.habitat_config.SIM_FREQ)
        for _ in range(steps):
            self.internal_step()
        self.gripper.update()
        self.robot.update()

    def get_state(self, include_robot=True):
        """Get the (kinematic) state of the simulation."""
        state = {
            "rigid_objs_T": {
                handle: obj.transformation
                for handle, obj in self.rigid_objs.items()
            },
            "art_objs_T": {
                handle: obj.transformation
                for handle, obj in self.art_objs.items()
            },
            "art_objs_qpos": {
                handle: obj.joint_positions
                for handle, obj in self.art_objs.items()
            },
        }
        if include_robot:
            state.update(
                {
                    "robot_state": self.robot.get_state(),
                    "grasped_obj": self.gripper.grasped_obj,
                    "grasped_marker": self.gripper.grasped_marker,
                }
            )
        return state

    def set_state(self, state: dict, include_robot=True):
        """Set the kinematic state of the simulation.

        Notes:
            The velocities and forces are set to 0.
            Be careful when using this function.
        """
        for handle, T in state["rigid_objs_T"].items():
            obj = self.rigid_objs[handle]
            obj.transformation = mn_utils.orthogonalize(T)
            obj.linear_velocity = mn.Vector3.zero_init()
            obj.angular_velocity = mn.Vector3.zero_init()

        for handle, T in state["art_objs_T"].items():
            art_obj = self.art_objs[handle]
            art_obj.transformation = mn_utils.orthogonalize(T)

        for handle, qpos in state["art_objs_qpos"].items():
            art_obj = self.art_objs[handle]
            art_obj.clear_joint_states()
            art_obj.joint_positions = qpos
            # art_obj.joint_velocities = np.zeros_like(art_obj.joint_velocities)
            # art_obj.joint_forces = np.zeros_like(art_obj.joint_forces)

        if include_robot:
            self.robot.set_state(state["robot_state"])

            self.gripper.desnap(True)  # desnap anyway
            if state["grasped_obj"] is not None:
                self.gripper.snap_to_obj(state["grasped_obj"])
            elif state["grasped_marker"] is not None:
                self.gripper.snap_to_marker(state["grasped_marker"])

    def sync_agent(self):
        """Synchronize the virtual agent with the robot.
        Thus, we can reuse habitat-baselines utilities for map.

        Notes:
            `habitat_sim.AgentState` uses np.quaternion (w, x, y, z) for rotation;
            however, it accepts a list of (x, y, z, w) as rvalue.
        """
        agent_state = self._default_agent.get_state()
        # agent_state.position = np.array(self.robot.sim_obj.translation)
        agent_state.position = self.robot.base_pos
        # align robot x-axis with agent z-axis
        agent_state.rotation = mn_utils.to_list(
            self.robot.sim_obj.rotation
            * mn.Quaternion.rotation(mn.Rad(-1.57), mn.Vector3(0, 1, 0))
        )
        self._default_agent.set_state(agent_state)

    def sync_pyb_robot(self):
        self.pyb_robot.set_joint_states(self.robot.arm_joint_pos)

    def step(self, action: Optional[int] = None):
        # virtual agent's action, only for compatibility.
        if action is not None:
            self._default_agent.act(action)

        if self.habitat_config.CONCUR_RENDER:
            self.gripper.update()
            self.robot.update()
            # self._prev_sim_obs = self.start_async_render()
            self._prev_sim_obs = self.get_sensor_observations()

        # step physics
        for _ in range(self.habitat_config.CONTROL_FREQ):
            self.internal_step()

        # sync virtual agent
        # self.sync_agent()
        self.sync_pyb_robot()

        if self.habitat_config.CONCUR_RENDER:
            # self._prev_sim_obs = self.get_sensor_observations_async_finish()
            return self._sensor_suite.get_observations(self._prev_sim_obs)

        return self.get_observations()

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def get_rigid_obj(self, index: int):
        handle = list(self.rigid_objs.keys())[index]
        return self.rigid_objs[handle]

    def get_rigid_objs_pos(self):
        """Get the positions of all rigid objects."""
        return np.stack(
            [
                np.array(obj.translation, dtype=np.float32)
                for obj in self.rigid_objs.values()
            ],
            axis=0,
        )

    def get_rigid_objs_pos_dict(self):
        return {
            k: np.array(obj.translation, dtype=np.float32)
            for k, obj in self.rigid_objs.items()
        }

    def get_target(self, index):
        handle = list(self.targets.keys())[index]
        tgt_obj = self.rigid_objs[handle]
        tgt_T = self.targets[handle]
        return tgt_obj, tgt_T

    def get_marker(self, index):
        # return list(self.markers.values())[index]
        return self.markers[index]

    def get_robot_collision(
        self, include_grasped_obj=True, link_ids=None, verbose=False
    ):
        robot_id = self.robot.object_id
        grasped_obj_id = self.gripper.grasped_obj_id
        contact_points = self.get_physics_contact_points()

        contact_infos = coll_utils.get_contact_infos(
            contact_points, robot_id, link_ids=link_ids
        )
        if include_grasped_obj and grasped_obj_id is not None:
            contact_infos.extend(
                coll_utils.get_contact_infos(contact_points, grasped_obj_id)
            )

        if len(contact_infos) > 0:
            max_force = max(x["normal_force"] for x in contact_infos)

            # -------------------------------------------------------------------------- #
            # DEBUG(jigu): too large force usually means that base has penetrated some obj.
            # -------------------------------------------------------------------------- #
            if verbose and max_force > 1e6:
                print(
                    "DEBUG (collision)",
                    self.habitat_config["EPISODE"]["episode_id"],
                    self.habitat_config["EPISODE"]["scene_id"],
                )
                for info in contact_infos:
                    # if info["normal_force"] < 1e3:
                    #     continue
                    print(
                        "collide with",
                        get_object_handle_by_id(self, info["object_id"]),
                        info,
                    )
            # -------------------------------------------------------------------------- #
        else:
            max_force = 0.0
        return max_force

    def set_joint_pos_by_motor(
        self, art_obj: ManagedBulletArticulatedObject, link_id, pos, dt
    ):
        art_obj.awake = True
        motor_id = art_utils.get_motor_id_by_link_id(art_obj, link_id)
        jms = JointMotorSettings(pos, 0.3, 0, 0.3, 0.5)
        if motor_id is not None:
            ori_jms = art_obj.get_joint_motor_settings(motor_id)
            art_obj.update_joint_motor(motor_id, jms)
            self.internal_step_by_time(dt)
            art_obj.update_joint_motor(motor_id, ori_jms)
        else:
            motor_id = art_obj.create_joint_motor(link_id, jms)
            self.internal_step_by_time(dt)
            art_obj.remove_joint_motor(motor_id)

        # NOTE(jigu): Simulate one step after motor gain changes.
        self.internal_step()

    def set_fridge_state_by_motor(self, angle, dt=0.6):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        self.set_joint_pos_by_motor(art_obj, 2, angle, dt=dt)

    def set_fridge_state(self, angle):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        art_utils.set_joint_pos(art_obj, [1], [angle])

    def get_fridge_state(self):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        return art_obj.joint_positions[1]

    def update_camera(self, sensor_name, cam2world: mn.Matrix4):
        agent_inv_T = self._default_agent.scene_node.transformation.inverted()
        sensor = self._sensors[sensor_name]._sensor_object
        sensor.node.transformation = mn_utils.orthogonalize(
            agent_inv_T @ cam2world
        )

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    def _remove_viz_objs(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for name, obj in self.viz_objs.items():
            assert obj.is_alive, name
            if self.verbose:
                print(
                    "Remove a vis object",
                    name,
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        self.viz_objs = OrderedDict()

    def add_viz_obj(
        self,
        position: mn.Vector3,
        scale=mn.Vector3(1, 1, 1),
        rotation: Optional[mn.Quaternion] = None,
        template_name="coord_frame",
    ):
        obj_attr_mgr = self.get_object_template_manager()
        rigid_obj_mgr = self.get_rigid_object_manager()

        # register a new template for visualization
        template = obj_attr_mgr.get_template_by_handle(
            obj_attr_mgr.get_template_handles(template_name)[0]
        )
        template.scale = scale
        template_id = obj_attr_mgr.register_template(
            template, f"viz_{template_name}"
        )

        viz_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        obj_utils.make_render_only(viz_obj)
        viz_obj.translation = position
        if rotation is not None:
            viz_obj.rotation = rotation
        return viz_obj

    def visualize_frame(self, name, T: mn.Matrix4, scale=1.0):
        assert name not in self.viz_objs, name
        self.viz_objs[name] = self.add_viz_obj(
            position=T.translation,
            scale=mn.Vector3(scale),
            rotation=mn_utils.mat3_to_quat(T.rotation()),
            template_name="coord_frame",
        )

    def visualize_arrow(self, name, position, orientation, scale=1.0):
        assert name not in self.viz_objs, name
        rotation = mn.Quaternion.rotation(
            mn.Rad(orientation), mn.Vector3(0, 1, 0)
        )
        self.viz_objs[name] = self.add_viz_obj(
            position=position,
            scale=mn.Vector3(scale),
            rotation=rotation,
            template_name="arrow",
        )

    def visualize_markers(self):
        for name, marker in self.markers.items():
            self.visualize_frame(name, marker.transformation, scale=0.15)

    def visualize_target(self, index):
        tgt_obj, tgt_T = self.get_target(index)
        obj_bb = obj_utils.get_aabb(tgt_obj)
        viz_obj = self.add_viz_obj(
            position=tgt_T.translation,
            scale=obj_bb.size() * 0.5,
            rotation=mn_utils.mat3_to_quat(tgt_T.rotation()),
            template_name="transform_box",
        )
        self.viz_objs[f"target.{index}"] = viz_obj

    def visualize_region(
        self,
        name,
        region: mn.Range2D,
        T: mn.Matrix4,
        height=None,
        template="region_green",
    ):
        center = mn.Vector3(region.center_x(), region.center_y(), 0.0)
        center = T.transform_point(center)
        if height is not None:
            center.y = height
        scale = mn.Vector3(region.size_x(), region.size_y(), 1.0)
        viz_obj = self.add_viz_obj(
            position=center,
            scale=scale,
            rotation=mn_utils.mat3_to_quat(T.rotation()),
            template_name=template,
        )
        self.viz_objs[name] = viz_obj

    def render(self, mode: str):
        """Render with additional debug info.
        Users can add more visualization to viz_objs before calling sim.render().
        """
        # self.visualize_frame("ee_frame", self.robot.ee_T, scale=0.15)
        rendered_frame = super().render(mode=mode)
        # Remove visualization in case polluate observations
        self._remove_viz_objs()
        return rendered_frame
