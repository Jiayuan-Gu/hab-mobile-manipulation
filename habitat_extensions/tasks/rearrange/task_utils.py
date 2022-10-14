from typing import Sequence

import magnum as mn
import numpy as np

from habitat_extensions.utils import map_utils
from habitat_extensions.utils.geo_utils import transform_points

from .sim import RearrangeSim
from .task import RearrangeTask


def random_choice(x: Sequence, rng: np.random.RandomState = np.random):
    assert len(x) > 0
    if len(x) == 1:
        return x[0]
    else:
        return x[rng.choice(len(x))]


def compute_start_state(sim: RearrangeSim, tgt_pos, init_start_pos=None):
    """Compute a start state given the target position.
    The start position is the nearest navigible point to the target,
    and the start orientation is towards the target.
    """
    if init_start_pos is None:
        start_pos = sim.pathfinder.snap_point(tgt_pos)
    else:
        start_pos = sim.pathfinder.snap_point(init_start_pos)
    assert not np.isnan(start_pos).any(), tgt_pos
    offset = tgt_pos - np.array(start_pos)
    start_ori = np.arctan2(-offset[2], offset[0])
    return start_pos, start_ori


def sample_random_start_state(
    sim: RearrangeSim,
    goal_pos,
    max_trials=1,
    rng: np.random.RandomState = np.random,
):
    assert sim.pathfinder.is_navigable(goal_pos), goal_pos
    for _ in range(max_trials):
        start_pos = sim.pathfinder.get_random_navigable_point()
        geo_dist = sim.geodesic_distance(start_pos, goal_pos)
        if np.isinf(geo_dist):
            continue
        start_ori = rng.uniform(0, np.pi * 2)
        return start_pos, start_ori


def sample_random_start_state_v1(
    sim: RearrangeSim,
    max_trials=1,
    rng: np.random.RandomState = np.random,
):
    """Sample a random start state (pos, ori) in the largest island."""
    for _ in range(max_trials):
        start_pos = sim.pathfinder.get_random_navigable_point()
        if not sim.is_at_larget_island(start_pos):
            continue
        start_ori = rng.uniform(0, np.pi * 2)
        return start_pos, start_ori


def sample_noisy_start_state(
    sim: RearrangeSim,
    start_pos,
    target_pos,
    pos_noise,
    ori_noise,
    pos_noise_thresh,
    ori_noise_thresh,
    max_trials=1,
    verbose=False,
    rng: np.random.RandomState = np.random,
):
    """Sample a start state with Gaussian noise,
    given an initial start position and a target position to face towards.
    """
    for _ in range(max_trials):
        start_pos_noise = rng.normal(0, pos_noise, [3])
        start_pos_noise[1] = 0.0  # no noise on y-axis (up)
        start_pos_noise = np.clip(
            start_pos_noise, -pos_noise_thresh, pos_noise_thresh
        )
        if verbose:
            print("start_pos_noise", start_pos_noise)
        noisy_start_pos = start_pos + start_pos_noise

        is_navigable = sim.pathfinder.is_navigable(noisy_start_pos)
        if not is_navigable:
            if verbose:
                print("Not navigable start position", start_pos_noise)
            continue

        start_ori_noise = rng.normal(0, ori_noise)
        start_ori_noise = np.clip(
            start_ori_noise, -ori_noise_thresh, ori_noise_thresh
        )
        if verbose:
            print("start_ori_noise", start_ori_noise)

        offset = target_pos - noisy_start_pos
        start_ori = np.arctan2(-offset[2], offset[0])
        noisy_start_ori = start_ori + start_ori_noise

        return noisy_start_pos, noisy_start_ori


def check_ik_feasibility(
    sim: RearrangeSim,
    goal_in_world,
    thresh,
    verbose=False,
):
    sim.sync_pyb_robot()
    goal_in_pbase = sim.robot.transform(goal_in_world, "world2pbase")
    qpos = sim.pyb_robot.IK(goal_in_pbase, max_iters=100)
    err = sim.pyb_robot.compute_IK_error(goal_in_pbase, qpos)
    if verbose:
        print("The error to reach {} with IK: {}".format(goal_in_world, err))
    return err <= thresh


def check_collision_free(sim: RearrangeSim, threshold, n_steps=12):
    # NOTE(jigu): If we use 1/120 sim_freq, 12 steps is about 0.1s
    for _ in range(n_steps):
        # -------------------------------------------------------------------------- #
        # DEBUG: for episode 26 (set_table_220322/train)
        # -------------------------------------------------------------------------- #
        # import cv2

        # sim.robot.update_cameras()
        # cv2.imshow("debug", sim.render("robot_third_rgb"))
        # cv2.waitKey(1)

        # art_obj = sim.art_objs["kitchen_counter_:0000"]
        # if art_obj.joint_positions[-1] < 0:
        #     print("art obj failure")
        # -------------------------------------------------------------------------- #
        sim.internal_step()
        collision_force = sim.get_robot_collision(True)
        if collision_force > threshold:
            return False
    return True


def check_start_state(
    sim: RearrangeSim,
    task: RearrangeTask,
    start_pos,
    start_ori,
    task_type,
    max_ik_error=None,
    max_collision_force=None,
    verbose=False,
):
    # Cache state before check
    state = sim.get_state()

    # Set start state
    sim.robot.base_pos = start_pos
    sim.robot.base_ori = start_ori

    if task_type == "pick":
        ik_goal = task.pick_goal
    elif task_type == "place":
        sim.gripper.desnap(True)
        # sim.gripper.snap_to_obj(task.tgt_obj)
        sim.gripper.snap_to_obj(task.tgt_obj.object_id)
        ik_goal = task.place_goal
    elif task_type == "nav":
        pass
    elif task_type in [
        "open_drawer",
        "close_drawer",
        "open_fridge",
        "close_fridge",
    ]:
        pass
    else:
        raise NotImplementedError(task_type)

    # check ik feasibility
    if max_ik_error is not None and not check_ik_feasibility(
        sim, ik_goal, max_ik_error, verbose=verbose
    ):
        sim.set_state(state)
        return False

    # check collision
    if max_collision_force is not None and not check_collision_free(
        sim, max_collision_force
    ):
        if verbose:
            print("The start state is not collision-free")
        sim.set_state(state)
        return False

    # restore old state
    sim.set_state(state)
    return True


def compute_start_positions_from_map(
    sim: RearrangeSim, goal, height, region_size, meters_per_pixel=0.05
):
    """Get candidates for start position (x, y, z) given the top-down map."""
    pathfinder = sim.pathfinder
    # 0-dim is z-axis, and 1-dim is x-axis
    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel,
        height=height,
    ).astype(np.uint8)
    top_down_map = np.ascontiguousarray(top_down_map)

    grid_resolution = top_down_map.shape
    lower_bound, upper_bound = pathfinder.get_bounds()
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_origin = [lower_bound[2], lower_bound[0]]

    coord = map_utils.to_grid(goal[[2, 0]], grid_origin, grid_size)
    window_size = np.ceil(region_size / meters_per_pixel).astype(int)
    lower, upper = map_utils.get_boundary(coord, window_size, grid_resolution)
    region = top_down_map[lower[0] : upper[0], lower[1] : upper[1]]

    coords = np.stack(np.nonzero(region), axis=-1)  # [?, 2]
    coords[:, 0] += lower[0]
    coords[:, 1] += lower[1]
    positions = map_utils.from_grid(coords, grid_origin, grid_size)

    # reorder into [x, y, z]
    positions = [[x, height, z] for (z, x) in positions]

    # -------------------------------------------------------------------------- #
    # Debug
    # -------------------------------------------------------------------------- #
    # import matplotlib.pyplot as plt
    # from habitat.utils.visualizations import maps

    # top_down_map[lower[0] : upper[0], lower[1] : upper[1]] = 4
    # top_down_map[coord[0], coord[1]] = 2
    # for z, x in coords:
    #     top_down_map[z, x] = 4
    # plt.imshow(maps.colorize_topdown_map(top_down_map))
    # plt.show()
    # plt.close("all")
    # -------------------------------------------------------------------------- #

    return positions


def compute_region_goals(
    sim: RearrangeSim,
    goal,
    height,
    region_size,
    meters_per_pixel=0.05,
    delta_size=0.1,
    max_region_size=5.0,
):
    while region_size < max_region_size:
        nav_goals = compute_start_positions_from_map(
            sim,
            goal,
            height,
            region_size=region_size,
            meters_per_pixel=meters_per_pixel,
        )
        if len(nav_goals) == 0:
            region_size = region_size + delta_size
            # print("search for a larger region_size", region_size)
            continue
        else:
            return nav_goals


def sample_navigable_point_within_region(
    sim: RearrangeSim,
    region: mn.Range2D,
    height,
    T: mn.Matrix4 = None,
    max_trials=100,
    rng: np.random.RandomState = np.random,
):
    """Sample a navigable point (x, z) within the region.
    The region is defined in xz-plane.
    """
    for _ in range(max_trials):
        xz = rng.uniform(region.min, region.max)
        pos = np.array([xz[0], 0, xz[1]])

        if T is not None:
            pos = np.array(T.transform_point(mn.Vector3(pos)))

        pos[1] = height

        if sim.pathfinder.is_navigable(pos):
            return pos


def get_navigable_positions(
    sim: RearrangeSim, height: float, meters_per_pixel=0.05
):
    pathfinder = sim.pathfinder
    # 0-dim is z-axis, and 1-dim is x-axis
    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel,
        height=height,
    ).astype(np.uint8)
    top_down_map = np.ascontiguousarray(top_down_map)

    # Get grid info
    grid_resolution = top_down_map.shape
    lower_bound, upper_bound = pathfinder.get_bounds()
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_origin = [lower_bound[2], lower_bound[0]]

    # Get navigable positions
    coords = np.stack(np.nonzero(top_down_map), axis=-1)  # [?, 2]
    zx = map_utils.from_grid(coords, grid_origin, grid_size)
    x, z = zx[:, 1], zx[:, 0]
    y = np.full_like(x, height)
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def visualize_positions_on_map(
    xyz, sim: RearrangeSim, height: float, meters_per_pixel=0.05
):
    pathfinder = sim.pathfinder
    # 0-dim is z-axis, and 1-dim is x-axis
    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel,
        height=height,
    ).astype(np.uint8)
    top_down_map = np.ascontiguousarray(top_down_map)

    # Get grid info
    grid_resolution = top_down_map.shape
    lower_bound, upper_bound = pathfinder.get_bounds()
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_origin = [lower_bound[2], lower_bound[0]]

    import cv2
    from habitat.utils.visualizations import maps

    coords = map_utils.to_grid(xyz[:, [2, 0]], grid_origin, grid_size)
    for z, x in coords:
        top_down_map[z, x] = 4
    cv2.imshow("map", maps.colorize_topdown_map(top_down_map))
    cv2.waitKey(0)


def filter_by_island_radius(
    sim: RearrangeSim, positions: np.ndarray, threshold=0.05
):
    positions = positions.astype(dtype=np.float32, copy=False)
    positions2 = []
    for p in positions:
        if not sim.pathfinder.is_navigable(p):
            # p2 will be a Vector3D if p is np.float64
            p2 = sim.pathfinder.snap_point(p)
            assert not np.isnan(p2).any(), p
            # assert np.linalg.norm(p2 - p) <= threshold, (p, p2)
            if np.linalg.norm(p2 - p) > threshold:
                raise RuntimeError((p, p2), sim._current_scene, sim.habitat_config.EPISODE["episode_id"])
            p = p2
        if sim.is_at_larget_island(p):
            positions2.append(p)
    return np.array(positions2, dtype=np.float32).reshape([-1, 3])


def compute_start_positions_from_map_v1(
    sim: RearrangeSim,
    T: mn.Matrix4,
    region: mn.Range2D,
    radius: float,
    height: float,
    meters_per_pixel=0.05,
    debug=False,
):
    """Get candidates for start position (x, y, z) given the top-down map."""
    xyz = get_navigable_positions(sim, height, meters_per_pixel)
    xyz_local = transform_points(xyz, np.array(T.inverted()))
    xz_local = xyz_local[:, [0, 2]]
    mask = np.ones([xyz.shape[0]], dtype=bool)

    if region is not None:
        mask2 = np.logical_and(xz_local >= region.min, xz_local <= region.max)
        mask2 = np.all(mask2, axis=-1)
        mask = np.logical_and(mask, mask2)

    if radius is not None:
        mask2 = np.linalg.norm(xz_local, axis=-1) <= radius
        mask = np.logical_and(mask, mask2)

    xyz = xyz[mask]
    xyz = filter_by_island_radius(sim, xyz, threshold=meters_per_pixel + 0.01)
    if debug:
        visualize_positions_on_map(xyz, sim, height, meters_per_pixel)
    return xyz


def compute_region_goals_v1(
    sim: RearrangeSim,
    T: mn.Matrix4,
    region: mn.Range2D,
    radius: float,
    height: float,
    meters_per_pixel=0.05,
    delta_size=0.1,
    max_radius=2.0,
    postprocessing=True,
    debug=False,
):
    xyz = get_navigable_positions(sim, height, meters_per_pixel)
    xyz_local = transform_points(xyz, np.array(T.inverted()))
    xz_local = xyz_local[:, [0, 2]]
    mask = np.ones([xyz.shape[0]], dtype=bool)
    if region is not None:
        mask2 = np.logical_and(xz_local >= region.min, xz_local <= region.max)
        mask2 = np.all(mask2, axis=-1)
        mask = np.logical_and(mask, mask2)

    while radius <= max_radius:
        mask3 = np.linalg.norm(xz_local, axis=-1) <= radius
        xyz2 = xyz[np.logical_and(mask, mask3)]
        if len(xyz2) == 0:
            radius += delta_size
            # print("search for a larger radius", radius)
            continue
        else:
            if postprocessing:
                xyz2 = filter_by_island_radius(
                    sim, xyz2, threshold=meters_per_pixel + 0.01
                )
            if len(xyz2) == 0:
                radius += delta_size
                # print("search for a larger radius", radius)
                continue

            if debug:
                visualize_positions_on_map(xyz2, sim, height, meters_per_pixel)
            return xyz2


def filter_positions(xyz: np.ndarray, T: mn.Matrix4, direction, clearance=0.0):
    xyz_local = transform_points(xyz, np.array(T.inverted()))  # [N, 3]
    direction = np.array(direction)
    sign = xyz_local @ direction
    return xyz[sign >= clearance]


def filter_positions_v1(
    xyz: np.ndarray, T: mn.Matrix4, region: mn.Range2D, radius: float
):
    xyz_local = transform_points(xyz, np.array(T.inverted()))
    xz_local = xyz_local[:, [0, 2]]
    mask = np.ones([xyz.shape[0]], dtype=bool)

    if region is not None:
        mask2 = np.logical_and(xz_local >= region.min, xz_local <= region.max)
        mask2 = np.all(mask2, axis=-1)
        mask = np.logical_and(mask, mask2)

    if radius is not None:
        mask2 = np.linalg.norm(xz_local, axis=-1) <= radius
        mask = np.logical_and(mask, mask2)

    xyz = xyz[mask]
    return xyz
