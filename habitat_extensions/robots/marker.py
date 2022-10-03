#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from habitat_sim.physics import ManagedBulletArticulatedObject


class Marker:
    """Data structure to track markers on articulated objects."""

    def __init__(
        self, uuid, art_obj: ManagedBulletArticulatedObject, link_id, offset
    ):
        self.uuid = uuid
        self.art_obj = art_obj
        self.link_id = link_id
        self.offset = offset

        self.pos_offset = art_obj.get_link_joint_pos_offset(link_id)
        self.link_node = art_obj.get_link_scene_node(link_id)

    @property
    def transformation(self):
        offset_T = mn.Matrix4.translation(mn.Vector3(self.offset))
        return self.link_node.transformation @ offset_T

    @property
    def pos(self):
        return np.array(self.transformation.translation, dtype=np.float32)

    @property
    def qpos(self):
        return self.art_obj.joint_positions[self.pos_offset]

    @property
    def qvel(self):
        return self.art_obj.joint_velocities[self.pos_offset]

    def set_semantic_id(self, semantic_id: int):
        for node in self.art_obj.get_link_visual_nodes(self.link_id):
            node.semantic_id = semantic_id

    # -------------------------------------------------------------------------- #
    # For challenge
    # -------------------------------------------------------------------------- #
    @property
    def offset_position(self):
        return self.offset

    @property
    def ao_parent(self):
        return self.art_obj
