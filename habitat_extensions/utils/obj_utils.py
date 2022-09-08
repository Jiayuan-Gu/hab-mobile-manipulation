import habitat_sim
import magnum as mn
from habitat_sim.physics import ManagedBulletRigidObject, MotionType


def get_aabb(obj: ManagedBulletRigidObject, transformed=False) -> mn.Range3D:
    """Get the axis-aligned bounding box of an object."""
    obj_node = obj.root_scene_node
    obj_bb = obj_node.cumulative_bb
    if transformed:
        obj_bb = habitat_sim.geo.get_transformed_bb(
            obj_node.cumulative_bb, obj_node.transformation
        )
    return obj_bb


def make_render_only(obj: ManagedBulletRigidObject):
    """Make the object only for rendering."""
    obj.motion_type = MotionType.KINEMATIC
    # obj.override_collision_group(
    #     habitat_sim.physics.CollisionGroups.Noncollidable
    # )
    obj.collidable = False
