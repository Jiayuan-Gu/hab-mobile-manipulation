import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.utils import art_utils

from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import (
    check_start_state,
    compute_region_goals_v1,
    compute_start_state,
    filter_by_island_radius,
    filter_positions,
    sample_noisy_start_state,
    visualize_positions_on_map,
)


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTask(RearrangeTask):
    # ---------------------------------------------------------------------------- #
    # Cache
    # ---------------------------------------------------------------------------- #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_start_states = dict()
        self._cache_start_positions = dict()
        self._cache_max_fridge_state = dict()

    def _has_cache_start_state(self, episode_id):
        if not self._config.get("USE_CACHE", False):
            return False
        key = (episode_id, self.tgt_idx)
        if key not in self._cache_start_states:
            return False
        return True

    def _get_cache_start_state(self, episode_id):
        # print("Cache is used", episode_id, self.tgt_idx)
        return self._cache_start_states[(episode_id, self.tgt_idx)]

    def _set_cache_start_state(self, episode_id, start_state):
        if not self._config.get("USE_CACHE", False):
            return
        key = (episode_id, self.tgt_idx)
        self._cache_start_states[key] = (
            start_state,
            self.start_ee_pos,
            self._sim.get_state(include_robot=False),
        )
        # print("Cache is set", episode_id, self.tgt_idx)

    def _has_cache_start_positions(self, episode_id):
        key = (episode_id, self.tgt_idx)
        if key not in self._cache_start_positions:
            return False
        return True

    def _get_cache_start_positions(self, episode_id):
        # print("Cache is used", episode_id, self.tgt_idx)
        return self._cache_start_positions[(episode_id, self.tgt_idx)]

    def _set_cache_start_positions(self, episode_id, start_positions):
        key = (episode_id, self.tgt_idx)
        self._cache_start_positions[key] = start_positions
        # print("Cache is set", episode_id, self.tgt_idx)

    def _get_max_fridge_state(self, scene_id):
        if scene_id not in self._cache_max_fridge_state:
            sim_state = self._sim.get_state()
            self._sim.set_fridge_state_by_motor(2.356)
            self._cache_max_fridge_state[
                scene_id
            ] = self._sim.get_fridge_state()
            self._sim.set_state(sim_state)
        return self._cache_max_fridge_state[scene_id]

    # ---------------------------------------------------------------------------- #
    # Actual initialization
    # ---------------------------------------------------------------------------- #
    def initialize(self, episode: RearrangeEpisode):
        start_state = None  # (start_pos, start_ori)
        sim_state = self._sim.get_state()  # snapshot

        # Recompute due to articulation
        self._maybe_recompute_navmesh(episode)

        n_targets = len(self._sim.targets)
        if "TARGET_INDEX" in self._config:
            tgt_indices = [self._config.TARGET_INDEX]
        else:
            tgt_indices = self.np_random.permutation(n_targets)

        # ---------------------------------------------------------------------------- #
        # Sample a collision-free start state
        # ---------------------------------------------------------------------------- #
        for tgt_idx in tgt_indices:
            self._set_target(tgt_idx)
            # NOTE(jigu): pick goal is defined before receptacle is set
            self.pick_goal = np.array(
                self.tgt_obj.translation, dtype=np.float32
            )

            if self._has_cache_start_state(episode.episode_id):
                (
                    start_state,
                    start_ee_pos,
                    start_sim_state,
                ) = self._get_cache_start_state(episode.episode_id)
                if start_state is not None:
                    self._initialize_ee_pos(start_ee_pos)
                    self._sim.set_state(start_sim_state, include_robot=False)
            else:
                self._initialize_target_receptacle(episode)
                start_state = self.sample_start_state(episode)
                self._set_cache_start_state(episode.episode_id, start_state)

            if start_state is not None:
                break
            else:
                logger.warning(
                    "Episode {}: fail to sample a valid start state for {}({})".format(
                        episode.episode_id, self.tgt_obj.handle, self.tgt_idx
                    )
                )
                self._sim.set_state(sim_state)  # recover from snapshot

        # -------------------------------------------------------------------------- #
        # Remove validation
        # -------------------------------------------------------------------------- #
        if start_state is None:
            self._set_target(tgt_indices[0])
            self.pick_goal = np.array(
                self.tgt_obj.translation, dtype=np.float32
            )
            self._initialize_target_receptacle(episode)

            start_state = self.sample_start_state(episode, no_validation=True)
            logger.warning(
                "Episode {}({}): sample a start state without validation".format(
                    episode.episode_id, episode.scene_id
                )
            )

        # Restore original navmesh
        self._maybe_restore_navmesh(episode)

        self._sim.robot.base_pos = start_state[0]
        self._sim.robot.base_ori = start_state[1]
        self._sim.internal_step_by_time(0.1)

        # -------------------------------------------------------------------------- #
        # Sanity check
        # -------------------------------------------------------------------------- #
        obj_pos = np.array(self.tgt_obj.translation, dtype=np.float32)
        err = np.linalg.norm(obj_pos - self.pick_goal)
        if err > self._err_thresh:
            logger.warning(
                "Episode {}/{}({}): pick goal err {} > {}".format(
                    episode.episode_id,
                    self.tgt_idx,
                    episode.scene_id,
                    err,
                    self._err_thresh,
                )
            )
            logger.info(
                "obj_pos: {}, pick_goal: {}".format(obj_pos, self.pick_goal)
            )
            # logger.info("initial_sim_state: {}".format(sim_state))
            # logger.info("current_sim_state: {}".format(self._sim.get_state()))
        # -------------------------------------------------------------------------- #

    def _initialize_target_receptacle(self, episode: RearrangeEpisode):
        receptacle_handle, receptacle_link_id = self.tgt_receptacle_info
        art_obj_mgr = self._sim.get_articulated_object_manager()

        self.tgt_receptacle = None
        self.tgt_receptacle_link = None
        self.init_start_pos = None
        self._err_thresh = 0.05
        self.pick_goal2 = None  # especially for drawer

        if self._has_target_in_fridge():
            self.tgt_receptacle = art_obj_mgr.get_object_by_handle(
                receptacle_handle
            )
            self.tgt_receptacle_link = self.tgt_receptacle.get_link_scene_node(
                receptacle_link_id
            )

            # print(art_utils.get_joints_info(self.tgt_receptacle))

            # init_range = self._config.get("FRIDGE_INIT_RANGE", [2.356, 2.356])
            # init_qpos = self.np_random.uniform(*init_range)

            # max_qpos = self._get_max_fridge_state(episode.scene_id)
            # init_qpos = np.clip(init_qpos, None, max_qpos)

            # Kinematic alternative to set link states
            # art_utils.set_joint_pos(self.tgt_receptacle, [1], [init_qpos])

            # Dynamic way to set link
            # self._sim.set_joint_pos_by_motor(
            #     self.tgt_receptacle, 2, init_qpos, dt=0.6
            # )
            # print(init_qpos, self.tgt_receptacle.joint_positions)

            T = self.tgt_receptacle.transformation
            offset = mn.Vector3(1.0, 0, 0)
            self.init_start_pos = np.array(T.transform_point(offset))
        elif self._has_target_in_drawer():
            self.tgt_receptacle = art_obj_mgr.get_object_by_handle(
                receptacle_handle
            )
            self.tgt_receptacle_link = self.tgt_receptacle.get_link_scene_node(
                receptacle_link_id
            )

            # init_range = self._config.get("DRAWER_INIT_RANGE", [0.5, 0.5])
            # init_qpos = self.np_random.uniform(*init_range)
            # self._err_thresh = init_qpos + 0.01

            # Dynamic way to set link
            # self._sim.set_joint_pos_by_motor(
            #     self.tgt_receptacle, receptacle_link_id, 0.5, dt=1.0
            # )

            # # Kinematic alternative to set link states
            # pos_offset = self.tgt_receptacle.get_link_joint_pos_offset(
            #     receptacle_link_id
            # )
            # T1 = self.tgt_receptacle_link.transformation
            # art_utils.set_joint_pos(
            #     self.tgt_receptacle, [pos_offset], [init_qpos]
            # )
            # T2 = self.tgt_receptacle_link.transformation
            # t = T2.translation - T1.translation
            # # Move object along with the drawer
            # # Assume a single object in the drawer
            # self.tgt_obj.translation = self.tgt_obj.translation + t
            self.pick_goal2 = np.array(
                self.tgt_obj.translation, dtype=np.float32
            )

            # # Generate some noise for obj in the drawer
            # obj_init_noise = self._config.get("OBJ_INIT_NOISE", 0.0)

            # if obj_init_noise > 0.0:
            #     # Add noise to move direction
            #     direction = t / np.linalg.norm(t)
            #     speed = self.np_random.randn() * obj_init_noise
            #     speed = np.clip(speed, -5.0, 5.0)

            #     # Add noise to orthogonal direction
            #     orth = mn.Matrix4.rotation_y(mn.Rad(np.pi / 2))
            #     direction2 = orth.transform_vector(direction)
            #     noise = self.np_random.randn() * obj_init_noise
            #     noise = np.clip(noise, -5.0, 5.0)
            #     self.tgt_obj.linear_velocity = (
            #         speed * direction + noise * direction2
            #     )
            #     t1 = self.tgt_obj.translation
            #     # self._sim.internal_step_by_time(1.0)
            #     self._sim.internal_step_by_time(0.6)
            #     t2 = self.tgt_obj.translation
            #     self._err_thresh += np.linalg.norm(t2 - t1)
            #     # print("obj noise", t2 - t1)
            #     # print(speed, self._err_thresh)

            T = self.tgt_receptacle_link.transformation
            offset = mn.Vector3(0.8, 0, 0)
            self.init_start_pos = np.array(T.transform_point(offset))

        # PrepareGroceries
        elif (
            self._config.get("FRIDGE_INIT", False)
            and len(self._sim.targets) == 3
        ):
            init_range = self._config.get("FRIDGE_INIT_RANGE", [2.356, 2.356])
            init_qpos = self.np_random.uniform(*init_range)

            max_qpos = self._get_max_fridge_state(episode.scene_id)
            init_qpos = np.clip(init_qpos, None, max_qpos)

            # self._sim.set_fridge_state_by_motor(init_qpos)
            self._sim.set_fridge_state(init_qpos)

    def sample_start_state(
        self,
        episode,
        max_trials=20,
        verbose=False,
        no_validation=False,
    ):
        # Generate a initial start pos
        start_pos, _ = compute_start_state(
            self._sim, self.pick_goal, init_start_pos=self.init_start_pos
        )

        # Skip if not in the largest island
        if not self._sim.is_at_larget_island(start_pos):
            logger.warning(
                "Episode {}: start_pos({}) is not at the largest island".format(
                    episode.episode_id, self.tgt_idx
                )
            )
            # return None

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        for i in range(max_trials):
            # Avoid extreme end-effector positions by resampling each time
            self._initialize_ee_pos()

            start_state = sample_noisy_start_state(
                self._sim,
                start_pos,
                self.pick_goal,
                pos_noise=pos_noise,
                ori_noise=ori_noise,
                pos_noise_thresh=2 * pos_noise,
                ori_noise_thresh=2 * ori_noise,
                max_trials=10,
                verbose=verbose,
                rng=self.np_random,
            )
            if start_state is None:
                continue
            if no_validation:
                return start_state
            max_ik_error = 0.14
            if self._has_target_in_drawer():
                max_ik_error = None
            is_valid = check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="pick",
                # max_ik_error=max_ik_error,
                max_ik_error=None,
                max_collision_force=0.0,
                verbose=verbose,
            )
            if is_valid:
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def render(self, mode):
        if self._has_target_in_container():
            # self._sim.visualize_frame(
            #     "receptacle", self.tgt_receptacle_link.transformation
            # )

            self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(
                self.pick_goal
            )

            # pos, ori = compute_start_state(
            #     self._sim, self.pick_goal, self.init_start_pos
            # )
            # self._sim.visualize_arrow("init_start_pos", pos, ori)

        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)
        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret


@registry.register_task(name="RearrangePickTask-v1")
class RearrangePickTaskV1(RearrangePickTask):
    def sample_start_state(
        self, episode, max_trials=20, verbose=False, no_validation=False
    ):
        # Use actual pick goal to compute start position
        # Especially when target is in drawer
        # pick_goal = np.array(self.tgt_obj.translation, dtype=np.float32)

        if self._has_cache_start_positions(episode.episode_id):
            start_positions = self._get_cache_start_positions(
                episode.episode_id
            )
        else:
            self._maybe_recompute_navmesh(episode, disable=False)
            pick_goal = (
                self.pick_goal if self.pick_goal2 is None else self.pick_goal
            )
            start_pos, _ = compute_start_state(self._sim, pick_goal)
            height = start_pos[1]
            # A hack to avoid stair
            if height > 0.2:
                height = 0.11094765
            T = mn.Matrix4.translation(pick_goal)
            # T = mn.Matrix4.translation(self.pick_goal)
            start_positions = compute_region_goals_v1(
                self._sim,
                T,
                region=None,
                radius=self._config.START_REGION_SIZE,
                height=height,
                max_radius=self._config.MAX_REGION_SIZE,
                postprocessing=False,
                debug=False,
            )
            self._maybe_restore_navmesh(episode, disable=False)
            # visualize_positions_on_map(start_positions, self._sim, height, 0.05)
            start_positions = filter_by_island_radius(
                self._sim, start_positions, threshold=0.5
            )
            # Post-processing for picking or placing in fridge
            if self._has_target_in_fridge():
                start_positions = filter_positions(
                    start_positions,
                    self._sim.markers["fridge_push_point"].transformation,
                    direction=[-1.0, 0.0, 0.0],
                    clearance=0.4,
                )
            # visualize_positions_on_map(start_positions, self._sim, height, 0.05)
            # NOTE(jigu): it is not accurate for drawer
            self._set_cache_start_positions(
                episode.episode_id, start_positions
            )

        # print(len(start_positions))
        if start_positions is None or len(start_positions) == 0:
            logger.warning(
                "Episode {} ({}): Unable to find any navigable point around the {}-th target given the map.".format(
                    episode.episode_id, episode.scene_id, self.tgt_idx
                )
            )
            return None

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        for i in range(max_trials):
            # Avoid extreme end-effector positions by resampling each time
            self._initialize_ee_pos()

            ind = self.np_random.choice(len(start_positions))
            start_state = sample_noisy_start_state(
                self._sim,
                start_positions[ind],
                # pick_goal,
                self.pick_goal,  # Note we use goal specification here!
                pos_noise=pos_noise,
                ori_noise=ori_noise,
                pos_noise_thresh=2 * pos_noise,
                ori_noise_thresh=2 * ori_noise,
                max_trials=10,
                verbose=verbose,
                rng=self.np_random,
            )
            if start_state is None:
                continue

            if check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="pick",
                max_collision_force=0.0,
                verbose=verbose,
            ):
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def _maybe_recompute_navmesh(
        self, episode: RearrangeEpisode, disable=True
    ):
        if disable:
            return
        super()._maybe_recompute_navmesh(episode)

    def _maybe_restore_navmesh(self, episode: RearrangeEpisode, disable=True):
        if disable:
            return
        super()._maybe_restore_navmesh(episode)
