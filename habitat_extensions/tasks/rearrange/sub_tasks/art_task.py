import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from ....robots.marker import Marker
from ..sim import RearrangeSim
from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import (
    check_collision_free,
    check_ik_feasibility,
    compute_start_state,
    filter_positions_v1,
    get_navigable_positions,
    sample_noisy_start_state,
)


class SetArticulatedObjectTask(RearrangeTask):
    marker: Marker
    marker_name: str
    tgt_qpos: float
    pick_goal: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_navigable_positions = dict()
        self._cache_max_fridge_state = dict()

    def initialize(self, episode: RearrangeEpisode):
        self._set_marker(episode)
        self._set_target_art_state()
        self._initialize_art_obj(episode)

        start_state = self.sample_start_state(
            episode, spawn_region=self._get_spawn_region()
        )

        if start_state is None:
            logger.warning(
                "Episode {}({}): use region2 for {}".format(
                    episode.episode_id, episode.scene_id, self.marker_name
                )
            )
            start_state = self.sample_start_state(
                episode, spawn_region=self._get_spawn_region2()
            )

        if start_state is None:
            raise RuntimeError(
                "Episode {}({}): fail to sample a valid start state for {}".format(
                    episode.episode_id, episode.scene_id, self.marker_name
                )
            )

        self._sim.robot.base_pos = start_state[0]
        self._sim.robot.base_ori = start_state[1]
        self._sim.internal_step_by_time(0.1)

    def _set_marker(self, episode):
        raise NotImplementedError

    def _get_spawn_region(self) -> mn.Range2D:
        return mn.Range2D(*self._config.SPAWN_REGION)

    def _get_spawn_region2(self):
        # In case the first spawn region is too crowded.
        raise NotImplementedError

    def _initialize_art_obj(self, episode: RearrangeEpisode):
        raise NotImplementedError

    def _set_target_art_state(self):
        raise NotImplementedError

    def _get_navigable_positions(self, scene_id):
        if scene_id not in self._cache_navigable_positions:
            return None
        else:
            return self._cache_navigable_positions[scene_id]

    def sample_start_state(
        self,
        episode: RearrangeEpisode,
        spawn_region: mn.Range2D,
        max_trials=20,
        max_ik_error=None,
        max_collision_force=0.0,
        verbose=False,
    ):
        navigable_positions = self._get_navigable_positions(episode.scene_id)
        if navigable_positions is None:
            start_pos, _ = compute_start_state(self._sim, self.marker.pos)
            height = start_pos[1]
            navigable_positions = get_navigable_positions(self._sim, height)
            self._cache_navigable_positions[
                episode.scene_id
            ] = navigable_positions

        # Assume x-forward and y-up
        if self._config.USE_MARKER_T:
            T = self.marker.transformation
            if "MARKER_REL_T" in self._config:
                rel_T = mn.Matrix4(np.array(self._config.MARKER_REL_T))
                T = T @ rel_T
        else:
            T = self.marker.art_obj.transformation

        # Filter start positions
        start_positions = filter_positions_v1(
            navigable_positions, T, spawn_region, radius=None
        )
        if len(start_positions) == 0:
            return None

        # NOTE(jigu): p-viz-plan first compute it and then set art state
        look_at_pos = np.array(T.translation, dtype=np.float32)

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        # Sample collision-free state
        for i in range(max_trials):
            self._initialize_ee_pos()

            ind = self.np_random.choice(len(start_positions))
            start_state = sample_noisy_start_state(
                self._sim,
                start_positions[ind],
                look_at_pos,
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

            sim_state = self._sim.get_state()  # snapshot
            self._sim.robot.base_pos = start_state[0]
            self._sim.robot.base_ori = start_state[1]

            if max_ik_error is not None and not check_ik_feasibility(
                self._sim, self.marker.pos, max_ik_error, verbose=verbose
            ):
                continue

            if max_collision_force is not None:
                is_safe = check_collision_free(self._sim, max_collision_force)
                if not is_safe:
                    self._sim.set_state(sim_state)  # restore
                    if verbose:
                        print("Not collision-free")
                    continue

            if verbose:
                print(f"Find a valid start state at {i}-th trial")

            self._sim.set_state(sim_state)  # restore state
            return start_state

    def render(self, mode):
        self._sim.visualize_frame(
            self.marker_name, self.marker.transformation, scale=0.15
        )
        # self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(self.pick_goal)
        ret = self._sim.render(mode)
        return ret


@registry.register_task(name="RearrangeOpenDrawerTask-v0")
class RearrangeOpenDrawerTask(SetArticulatedObjectTask):
    def _set_marker(self, episode: RearrangeEpisode):
        self.tgt_obj, self.tgt_T = self._sim.get_target(0)
        self.pick_goal = np.array(self.tgt_obj.translation, dtype=np.float32)

        # Find corresponding marker given target receptacle info
        receptacle_handle, receptacle_link_id = episode.target_receptacles[0]
        assert "kitchen_counter" in receptacle_handle, receptacle_handle
        self.marker_name = "cab_push_point_{}".format(receptacle_link_id)
        self.marker = self._sim.markers[self.marker_name]

    def _get_spawn_region2(self):
        return mn.Range2D([0.8, -0.6], [1.5, 0.6])

    def _initialize_art_obj(self, episode: RearrangeEpisode):
        self.marker.art_obj.clear_joint_states()
        self.marker.art_obj.joint_positions = np.zeros([7])

    def _set_target_art_state(self):
        self.tgt_qpos = self._config.get("TARGET_ART_STATE", 0.45)


@registry.register_task(name="RearrangeOpenFridgeTask-v0")
class RearrangeOpenFridgeTask(SetArticulatedObjectTask):
    def _set_marker(self, episode: RearrangeEpisode):
        self.tgt_obj, self.tgt_T = self._sim.get_target(1)
        self.pick_goal = np.array(self.tgt_obj.translation, dtype=np.float32)

        # Find corresponding marker given target receptacle info
        receptacle_handle, receptacle_link_id = episode.target_receptacles[1]
        assert (
            "frige" in receptacle_handle or "fridge" in receptacle_handle
        ), receptacle_handle
        self.marker_name = "fridge_push_point"
        self.marker = self._sim.markers[self.marker_name]

    def _get_spawn_region2(self):
        return mn.Range2D([0.8, -1.0], [2.0, 1.0])

    def _initialize_art_obj(self, episode: RearrangeEpisode):
        self.marker.art_obj.clear_joint_states()
        self.marker.art_obj.joint_positions = np.zeros([2])

    def _set_target_art_state(self):
        self.tgt_qpos = self._config.get("TARGET_ART_STATE", np.pi / 2)


@registry.register_task(name="RearrangeCloseDrawerTask-v0")
class RearrangeCloseDrawerTask(RearrangeOpenDrawerTask):
    def _set_marker(self, episode: RearrangeEpisode):
        super()._set_marker(episode)
        self.tgt_obj.transformation = self.tgt_T

    def _get_spawn_region2(self):
        return mn.Range2D([0.3, -0.6], [1.0, 0.6])

    def _initialize_art_obj(self, episode: RearrangeEpisode):
        n_qpos = 7
        init_qpos = np.zeros((n_qpos,))
        n_open = self.np_random.randint(n_qpos)
        if n_open > 0:
            open_idxs = self.np_random.choice(n_qpos, n_open)
            init_qpos[open_idxs] = self.np_random.uniform(
                0.0, 0.1, size=n_open
            )
        init_qpos[self.marker.pos_offset] = self.np_random.uniform(0.4, 0.5)

        self.marker.art_obj.clear_joint_states()
        self.marker.art_obj.joint_positions = init_qpos

    def _set_target_art_state(self):
        self.tgt_qpos = 0.0


@registry.register_task(name="RearrangeCloseFridgeTask-v0")
class RearrangeCloseFridgeTask(RearrangeOpenFridgeTask):
    def _set_marker(self, episode: RearrangeEpisode):
        super()._set_marker(episode)
        self.tgt_obj.transformation = self.tgt_T

    def _initialize_art_obj(self, episode: RearrangeEpisode):
        init_qpos = self.np_random.uniform(np.pi / 2 - 0.15, 2.356)

        max_qpos = self._get_max_fridge_state(episode.scene_id)
        init_qpos = np.clip(init_qpos, None, max_qpos)

        self.marker.art_obj.clear_joint_states()
        # self._sim.set_fridge_state_by_motor(init_qpos)
        self._sim.set_fridge_state(init_qpos)

    def _set_target_art_state(self):
        self.tgt_qpos = 0.0

    def _get_max_fridge_state(self, scene_id):
        if scene_id not in self._cache_max_fridge_state:
            sim_state = self._sim.get_state()
            self._sim.set_fridge_state_by_motor(2.356)
            self._cache_max_fridge_state[
                scene_id
            ] = self._sim.get_fridge_state()
            self._sim.set_state(sim_state)
        return self._cache_max_fridge_state[scene_id]
