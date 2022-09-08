import magnum as mn
import numpy as np
from habitat.core.simulator import Simulator
from habitat_sim.physics import (
    CollisionGroupHelper,
    CollisionGroups,
    ManagedBulletArticulatedObject,
    ManagedBulletRigidObject,
    RigidConstraintSettings,
    RigidConstraintType,
)

from habitat_extensions.robots.base_robot import Robot
from habitat_extensions.robots.marker import Marker


class MagicGripper:
    def __init__(self, sim: Simulator, robot: Robot):
        self._sim = sim
        self._robot = robot

        self._grasped_obj = None
        self._grasped_marker = None
        self._constraint_ids = None

    def reconfigure(self):
        self._grasped_obj = None
        self._grasped_marker = None
        self._constraint_ids = None

        # It seems to only affect objects created after it.
        CollisionGroupHelper.set_mask_for_group(
            CollisionGroups.UserGroup7, ~CollisionGroups.Robot
        )

    def reset(self):
        self.desnap(True)

    def create_p2p_constraint(
        self,
        pivot_in_link,
        pivot_in_obj,
        obj,
        link_id=None,
        max_impulse=1000.0,
    ):
        c = RigidConstraintSettings()
        c.constraint_type = RigidConstraintType.PointToPoint
        c.object_id_a = self._robot.object_id
        c.link_id_a = self._robot.params.ee_link
        c.object_id_b = obj.object_id
        if link_id is not None:
            c.link_id_b = link_id
        c.pivot_a = pivot_in_link
        c.pivot_b = pivot_in_obj
        c.max_impulse = max_impulse
        return self._sim.create_rigid_constraint(c)

    def snap_to_obj(self, obj: ManagedBulletRigidObject, force=True):
        assert isinstance(obj, ManagedBulletRigidObject), type(obj)

        # Already grasp
        if self.grasped_obj_id == obj.object_id:
            return

        if self.is_grasped:
            raise RuntimeError("Tried to snap to obj {}.".format(obj.handle))

        if force:
            # kinematically set the object in the hand
            gripper_T = self._robot.gripper_T
            obj.transformation = gripper_T

        self._grasped_obj = obj
        # remove collision between grasped object and robot
        self._grasped_obj.override_collision_group(CollisionGroups.UserGroup7)

        # create constraints (decide axes)
        self._constraint_ids = [
            self.create_p2p_constraint(
                mn.Vector3(0.1, 0, 0), mn.Vector3(0, 0, 0), obj
            ),
            self.create_p2p_constraint(
                mn.Vector3(0.0, 0, 0), mn.Vector3(-0.1, 0, 0), obj
            ),
            self.create_p2p_constraint(
                mn.Vector3(0.1, 0.0, 0.1), mn.Vector3(0.0, 0.0, 0.1), obj
            ),
        ]
        if any((x == -1 for x in self._constraint_ids)):
            raise RuntimeError("Bad constraint")

        # print("snap to obj", self._grasped_obj.object_id, self._constraint_ids)

    def snap_to_marker(self, marker: Marker):
        assert isinstance(marker, Marker), type(marker)

        # Already grasp
        if self.grasped_marker_id == marker.uuid:
            return

        if self.is_grasped:
            raise RuntimeError(
                "Tried to snap to marker {}.".format(marker.uuid)
            )

        self._grasped_marker = marker

        # NOTE(jigu): params from p-viz-plan, but change grasp point
        self._constraint_ids = [
            self.create_p2p_constraint(
                mn.Vector3(0.1, 0, 0),
                mn.Vector3(marker.offset),
                marker.art_obj,
                marker.link_id,
                max_impulse=100,
            )
        ]
        if any((x == -1 for x in self._constraint_ids)):
            raise RuntimeError("Bad constraint")

    def desnap(self, override_collision=False):
        if not self.is_grasped:
            return

        # print(
        #     "desnap",
        #     self.grasped_obj_id,
        #     self.grasped_marker_id,
        #     self._constraint_ids,
        # )

        if self._grasped_obj is not None:
            # NOTE(jigu): If the collision group is reverted to default,
            # there will be collision between the object and robot instantly.
            # It seems that the object is still "attached", as collision needs to be resolved.
            if override_collision:
                self._grasped_obj.override_collision_group(
                    CollisionGroups.Default
                )

        self._grasped_obj = None
        self._grasped_marker = None

        for constraint_id in self._constraint_ids:
            self._sim.remove_rigid_constraint(constraint_id)
        self._constraint_ids = None

    @property
    def is_grasped(self):
        return (
            self._grasped_obj is not None or self._grasped_marker is not None
        )

    @property
    def grasped_obj(self):
        return self._grasped_obj

    @property
    def grasped_obj_id(self):
        if self._grasped_obj is None:
            return None
        else:
            return self._grasped_obj.object_id

    @property
    def grasped_marker(self):
        return self._grasped_marker

    @property
    def grasped_marker_id(self):
        if self._grasped_marker is None:
            return None
        else:
            return self._grasped_marker.uuid

    def is_invalid_grasp(self, threshold):
        if not self.is_grasped:
            return False

        if self._grasped_obj is not None:
            gripper_pos = self._robot.gripper_pos
            obj_pos = np.array(self._grasped_obj.translation, dtype=np.float32)
            dist = np.linalg.norm(gripper_pos - obj_pos)
            # print("invalid grasp (obj):", dist)
            return dist > threshold

        if self._grasped_marker is not None:
            gripper_pos = self._robot.gripper_pos
            marker_pos = self._grasped_marker.pos
            dist = np.linalg.norm(gripper_pos - marker_pos)
            # print("invalid grasp (marker):", dist)
            return dist > threshold
