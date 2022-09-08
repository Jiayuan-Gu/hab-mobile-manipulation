"""Utilities for articulations.

Notes:
    The id of the root link is -1.
    The root link id is not included in get_link_ids().
"""
from typing import Dict, List, Union

import numpy as np
from habitat_sim.physics import (
    JointMotorSettings,
    ManagedBulletArticulatedObject,
)


# ---------------------------------------------------------------------------- #
# Link-joint mapping
# ---------------------------------------------------------------------------- #
def get_link_joint_mapping(art_obj: ManagedBulletArticulatedObject):
    link_joint_pos_offset: Dict[int, int] = {}
    link_num_joint_pos: Dict[int, int] = {}
    link_dof_offset: Dict[int, int] = {}
    link_num_dofs: Dict[int, int] = {}
    for link_id in art_obj.get_link_ids():
        link_joint_pos_offset[link_id] = art_obj.get_link_joint_pos_offset(
            link_id
        )
        link_num_joint_pos[link_id] = art_obj.get_link_num_joint_pos(link_id)
        link_dof_offset[link_id] = art_obj.get_link_dof_offset(link_id)
        link_num_dofs[link_id] = art_obj.get_link_num_dofs(link_id)
    return (
        link_joint_pos_offset,
        link_num_joint_pos,
        link_dof_offset,
        link_num_dofs,
    )


def get_joint_pos_offset(
    link_joint_pos_offset: Dict[int, int],
    link_num_joint_pos: Dict[int, int],
    link_ids: List[int],
):
    joint_pos_offset = []
    for link_id in link_ids:
        pos_offset = link_joint_pos_offset[link_id]
        num_pos = link_num_joint_pos[link_id]
        joint_pos_offset.extend(list(range(pos_offset, pos_offset + num_pos)))
    return joint_pos_offset


def get_link_id_by_name(art_obj: ManagedBulletArticulatedObject, name: str):
    for link_id in art_obj.get_link_ids():
        link_name = art_obj.get_link_name(link_id)
        if link_name == name:
            return link_id


# ---------------------------------------------------------------------------- #
# Joint states
# ---------------------------------------------------------------------------- #
def set_joint_pos(
    art_obj: ManagedBulletArticulatedObject,
    pos_offset: List[int],
    pos: Union[List[float], np.ndarray],
):
    assert len(pos_offset) == len(pos)
    joint_positions: List[float] = art_obj.joint_positions
    for i, offset in enumerate(pos_offset):
        joint_positions[offset] = pos[i]
    art_obj.joint_positions = joint_positions


def set_joint_force(
    art_obj: ManagedBulletArticulatedObject,
    dof_offset: List[int],
    force: Union[List[float], np.ndarray],
):
    assert len(dof_offset) == len(force)
    joint_forces: List[float] = art_obj.joint_forces
    for i, offset in enumerate(dof_offset):
        joint_forces[offset] = force[i]
    art_obj.joint_forces = joint_forces


# ---------------------------------------------------------------------------- #
# Joint motor
# ---------------------------------------------------------------------------- #
def remove_existing_joint_motors(art_obj: ManagedBulletArticulatedObject):
    for motor_id in art_obj.existing_joint_motor_ids:
        art_obj.remove_joint_motor(motor_id)


def create_all_motors(
    art_obj: ManagedBulletArticulatedObject,
    pos=0.0,
    pos_gain=0.0,
    vel=0.0,
    vel_gain=0.0,
    max_impulse=1000.0,  # default value
):
    """Create linear motors for all joints"""
    jms = JointMotorSettings(pos, pos_gain, vel, vel_gain, max_impulse)
    art_obj.create_all_motors(jms)


def set_motor_pos(
    art_obj: ManagedBulletArticulatedObject,
    motor_ids: List[int],
    position_targets: Union[List[float], np.ndarray],
):
    for i, motor_id in enumerate(motor_ids):
        update_motor(art_obj, motor_id, position_target=position_targets[i])


def set_motor_vel(
    art_obj: ManagedBulletArticulatedObject,
    motor_ids: List[int],
    velocity_targets: Union[List[float], np.ndarray],
):
    assert len(motor_ids) == len(velocity_targets), (
        motor_ids,
        velocity_targets,
    )
    for i, motor_id in enumerate(motor_ids):
        update_motor(art_obj, motor_id, velocity_target=velocity_targets[i])


def get_motor_vel(
    art_obj: ManagedBulletArticulatedObject,
    motor_ids: List[int],
):
    velocity_targets = []
    for motor_id in motor_ids:
        jms = art_obj.get_joint_motor_settings(motor_id)
        velocity_targets.append(jms.velocity_target)
    return velocity_targets


def update_motor(
    art_obj: ManagedBulletArticulatedObject, motor_id: int, **kwargs
):
    if len(kwargs) == 0:
        return
    jms = art_obj.get_joint_motor_settings(motor_id)
    for k, v in kwargs.items():
        setattr(jms, k, v)
    art_obj.update_joint_motor(motor_id, jms)
    return jms


def get_motor_id_by_link_id(art_obj, link_id):
    for (motor_id, _link_id) in art_obj.existing_joint_motor_ids.items():
        if link_id == _link_id:
            return motor_id
    return None


# ---------------------------------------------------------------------------- #
# Info
# ---------------------------------------------------------------------------- #
def get_links_info(art_obj: ManagedBulletArticulatedObject):
    info_str = []
    for link_id in [-1] + art_obj.get_link_ids():
        link_name = art_obj.get_link_name(link_id)
        sub_info_str = f"{link_name} (link_id={link_id})"
        info_str.append(sub_info_str)
    info_str = "\n".join(info_str)
    return info_str


def get_joints_info(art_obj: ManagedBulletArticulatedObject):
    info_str = []
    pos_limits = art_obj.joint_position_limits
    for link_id in art_obj.get_link_ids():
        link_name = art_obj.get_link_name(link_id)
        pos_offset = art_obj.get_link_joint_pos_offset(link_id)
        num_pos = art_obj.get_link_num_joint_pos(link_id)
        joint_name = art_obj.get_link_joint_name(link_id)
        joint_type = str(art_obj.get_link_joint_type(link_id))
        low = pos_limits[0][pos_offset : pos_offset + num_pos]
        high = pos_limits[1][pos_offset : pos_offset + num_pos]
        joint_str = (
            f"{joint_name} (link_id={link_id}, link_name={link_name}): "
        )
        joint_str += f"type={joint_type}, pos_offset={pos_offset}, num_pos={num_pos}, limits=[{low}, {high}]"
        info_str.append(joint_str)
    return "\n".join(info_str)


def jms2str(jms: JointMotorSettings):
    return ",".join(
        [
            "{:s}={:s}".format(k, str(getattr(jms, k)))
            for k in [
                "motor_type",
                "position_target",
                "position_gain",
                "velocity_target",
                "velocity_gain",
                "max_impulse",
            ]
        ]
    )


def get_joint_motors_info(art_obj: ManagedBulletArticulatedObject):
    info_str = []
    for motor_id, link_id in art_obj.existing_joint_motor_ids.items():
        joint_name = art_obj.get_link_joint_name(link_id)
        jms = art_obj.get_joint_motor_settings(motor_id)
        jms_str = jms2str(jms)
        setting_str = (
            f"{joint_name} (motor_id={motor_id}, link_id={link_id}): {jms_str}"
        )
        info_str.append(setting_str)
    info_str = "\n".join(info_str)
    return info_str
