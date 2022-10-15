import copy
import gzip
import json
import os
from typing import Dict, List, Optional, Tuple

import attr
import habitat_sim
import magnum as mn
import numpy as np
from habitat import logger
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator

from .sim import RearrangeSim


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
    r"""Specifies additional objects, targets, markers, and ArticulatedObject states for a particular instance of an object rearrangement task.

    :property ao_states: Lists modified ArticulatedObject states for the scene: {instance_handle -> {link, state}}
    :property rigid_objs: A list of objects to add to the scene, each with: (handle, transform)
    :property targets: Maps an object instance to a new target location for placement in the task. {instance_name -> target_transform}
    :property markers: Indicate points of interest in the scene such as grasp points like handles. {marker name -> (type, (params))}
    """
    ao_states: Dict[str, Dict[int, float]]
    rigid_objs: List[Tuple[str, np.ndarray]]
    targets: Dict[str, np.ndarray]
    markers: List[Dict]
    target_receptacles: List[Tuple[str, int]]
    goal_receptacles: List[Tuple[str, int]]
    name_to_receptacle: Dict[str, str] = attr.ib(factory=dict)

    # path to the SceneDataset config file
    scene_dataset_config: str = attr.ib(
        default="default", validator=not_none_validator
    )
    # list of paths to search for object config files in addition to the SceneDataset
    additional_obj_config_paths: List[str] = attr.ib(
        default=[], validator=not_none_validator
    )


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDataset(Dataset):
    episodes: List[RearrangeEpisode]

    def __init__(self, config: Optional[Config] = None):
        self.episodes = []
        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Keep provided episodes only
        episode_ids = config.get("EPISODE_IDS", [])
        if len(episode_ids) > 0:
            episode_ids = [str(x) for x in episode_ids]
            filter_fn = lambda x: x.episode_id in episode_ids
            self.episodes = list(filter(filter_fn, self.episodes))
            assert len(episode_ids) == len(self.episodes)

        num_episodes = config.get("NUM_EPISODES", -1)
        start = config.get("EPISODE_START", 0)
        end = None if num_episodes < 0 else (start + num_episodes)
        self.episodes = self.episodes[start:end]

    def from_json(self, json_str: str, scenes_dir: Optional[str]) -> None:
        deserialized = json.loads(json_str)
        for episode in deserialized["episodes"]:
            episode = RearrangeEpisode(**episode)
            self.episodes.append(episode)

    @property
    def episode_ids(self):
        return [x.episode_id for x in self.episodes]


@registry.register_task(name="RearrangeTask-v0")
class RearrangeTask(EmbodiedTask):
    _sim: RearrangeSim
    _is_episode_truncated: bool
    # should be called for force termination only
    _should_terminate: bool

    def overwrite_sim_config(self, sim_config, episode: RearrangeEpisode):
        sim_config.defrost()
        sim_config.SCENE = episode.scene_id
        # sim_config.SCENE_DATASET = episode.scene_dataset_config

        # To use baked lighting
        if self._config.get("USE_BAKED_SCENES", False):
            sim_config.SCENE = episode.scene_id.replace(
                "replica_cad", "replica_cad_baked_lighting"
            ).replace("v3", "Baked")
            sim_config.SCENE_DATASET = "data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"

        # Make a copy to avoid in-place modification
        sim_config["EPISODE"] = copy.deepcopy(episode.__dict__)
        # Initialize out of the room, so that it will not affect others
        sim_config["EPISODE"]["start_position"] = [50.0, 0, 50.0]
        sim_config.freeze()
        return sim_config

    def _check_episode_is_active(self, *args, **kwargs) -> bool:
        # NOTE(jigu): Be careful when you use this function to terminate.
        # It is called in task.step() after observations are updated.
        # task.step() is called in env.step() before measurements are updated.
        # return True
        return not self._should_terminate

    @property
    def is_episode_truncated(self):
        return self._is_episode_truncated

    @property
    def should_terminate(self):
        return self._should_terminate

    def seed(self, seed: int) -> None:
        # NOTE(jigu): Env will set the seed for random and np.random
        # when initializing episode iterator.
        self.np_random = np.random.RandomState(seed)

    def reset(self, episode: RearrangeEpisode):
        self._sim.reset()

        # Clear and cache
        self.tgt_idx = None
        self.tgt_obj, self.tgt_T = None, None
        self.tgt_receptacle_info = None
        self.start_ee_pos = None  # for ee-space controller
        self._target_receptacles = episode.target_receptacles
        self._goal_receptacles = episode.goal_receptacles

        self.initialize(episode)
        self._reset_stats()

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)
        return self._get_observations(episode)

    def initialize(self, episode: RearrangeEpisode):
        self._initialize_ee_pos()
        start_pos = self._sim.pathfinder.get_random_navigable_point()
        self._sim.robot.base_pos = start_pos
        self._sim.robot.base_ori = self.np_random.uniform(0, 2 * np.pi)
        self._sim.internal_step_by_time(0.1)

    def _get_start_ee_pos(self):
        # # NOTE(jigu): defined in pybullet link frame
        # start_ee_pos = np.array(
        #     self._config.get("START_EE_POS", [0.5, 0.0, 1.0]),
        #     dtype=np.float32,
        # )

        self._sim.pyb_robot.set_joint_states(
            self._sim.robot.params.arm_init_params
        )
        start_ee_pos = self._sim.pyb_robot.ee_state[4]

        # The noise can not be too large (e.g. 0.05)
        ee_noise = self._config.get("EE_NOISE", 0.025)
        if ee_noise > 0:
            noise = self.np_random.normal(0, ee_noise, [3])
            noise = np.clip(noise, -ee_noise * 2, ee_noise * 2)
            start_ee_pos = start_ee_pos + noise

        return np.float32(start_ee_pos)

    # def _initialize_ee_pos(self, start_ee_pos=None):
    #     """Initialize end-effector position."""
    #     if start_ee_pos is None:
    #         start_ee_pos = self._get_start_ee_pos()

    #     # print("start_ee_pos", start_ee_pos)
    #     self.start_ee_pos = start_ee_pos
    #     self._sim.robot.reset_arm()
    #     self._sim.sync_pyb_robot()
    #     arm_tgt_qpos = self._sim.pyb_robot.IK(self.start_ee_pos, max_iters=100)
    #     # err = self._sim.pyb_robot.compute_IK_error(start_ee_pos, arm_tgt_qpos)
    #     self._sim.robot.arm_joint_pos = arm_tgt_qpos
    #     self._sim.robot.arm_motor_pos = arm_tgt_qpos

    def _initialize_ee_pos(self, start_ee_pos=None):
        self._sim.robot.reset_arm()
        noise = self.np_random.normal(
            0, 0.05, size=len(self._sim.robot.arm_joint_pos)
        )
        arm_tgt_qpos = self._sim.robot.arm_joint_pos + np.clip(
            noise, -0.1, 0.1
        )
        self._sim.robot.arm_joint_pos = arm_tgt_qpos
        self._sim.robot.arm_motor_pos = arm_tgt_qpos

        self._sim.sync_pyb_robot()
        self.start_ee_pos = self._sim.pyb_robot.ee_state[4]
        # print("ee_pos", self._sim.robot.base_T.inverted().transform_point(self._sim.robot.gripper_pos))

    def _reset_stats(self):
        # NOTE(jigu): _is_episode_active is on-the-fly set in super().step()
        self._is_episode_active = True
        self._is_episode_truncated = False
        self._should_terminate = False

        # Record the initial robot pose for episodic sensors
        self.start_base_T = self._sim.robot.base_T
        # habitat frame
        self.resting_position = np.array(
            self._config.get("RESTING_POSITION", [0.5, 1.0, 0.0]),
            dtype=np.float32,
        )

    def _get_observations(self, episode):
        observations = self._sim.get_observations()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )
        return observations

    def render(self, mode):
        n_targets = len(self._sim.targets)

        for i in range(n_targets):
            tgt_obj, _ = self._sim.get_target(i)
            self._sim.set_object_bb_draw(True, tgt_obj.object_id)
            self._sim.visualize_target(i)
        self._sim.visualize_markers()

        ret = self._sim.render(mode)

        for i in range(n_targets):
            tgt_obj, _ = self._sim.get_target(i)
            self._sim.set_object_bb_draw(False, tgt_obj.object_id)

        return ret

    # -------------------------------------------------------------------------- #
    # Targets
    # -------------------------------------------------------------------------- #
    tgt_idx: int
    tgt_obj: habitat_sim.physics.ManagedBulletRigidObject
    tgt_T: mn.Matrix4
    tgt_receptacle_info: Tuple[str, int]

    def _set_target(self, index):
        self.tgt_idx = index
        self.tgt_obj, self.tgt_T = self._sim.get_target(self.tgt_idx)
        self.tgt_receptacle_info = self._target_receptacles[self.tgt_idx]

    def _has_target_in_drawer(self):
        receptacle_handle, receptacle_link_id = self.tgt_receptacle_info
        if receptacle_handle is None:  # for baked scenes
            return False
        if "kitchen_counter" in receptacle_handle and receptacle_link_id != 0:
            return True
        else:
            return False

    def _has_target_in_fridge(self):
        receptacle_handle, _ = self.tgt_receptacle_info
        if receptacle_handle is None:  # for baked scenes
            return False
        if "frige" in receptacle_handle or "fridge" in receptacle_handle:
            return True
        else:
            return False

    def _has_target_in_container(self):
        return self._has_target_in_drawer() or self._has_target_in_fridge()

    # -------------------------------------------------------------------------- #
    # Navmesh
    # -------------------------------------------------------------------------- #
    def _maybe_recompute_navmesh(self, episode: RearrangeEpisode):
        _recompute_navmesh = False
        for ao_state in episode.ao_states.values():
            if any(x > 0.0 for x in ao_state.values()):
                _recompute_navmesh = True
                break
        if _recompute_navmesh:
            self._sim._recompute_navmesh()
        self._recompute_navmesh = _recompute_navmesh

    def _maybe_restore_navmesh(self, episode: RearrangeEpisode):
        if not self._recompute_navmesh:
            return
        navmesh_path = episode.scene_id.replace("configs/scenes", "navmeshes")
        navmesh_path = navmesh_path.replace("scene_instance.json", "navmesh")
        self._sim.pathfinder.load_nav_mesh(navmesh_path)
        self._sim._cache_largest_island()
        self._recompute_navmesh = False

    def _check_art_abnormal(self, episode: RearrangeEpisode):
        art_obj_mgr = self._sim.get_articulated_object_manager()
        flag = True
        for ao_handle, ao_state in episode.ao_states.items():
            art_obj = art_obj_mgr.get_object_by_handle(ao_handle)
            qpos = art_obj.joint_positions
            for link_id, joint_state in ao_state.items():
                pos_offset = art_obj.get_link_joint_pos_offset(int(link_id))
                if np.abs(qpos[pos_offset] - joint_state) > 0.05:
                    flag = True
                    break

        if flag:
            logger.info(
                "Episode {}({}): detected abnormal".format(
                    episode.episode_id, episode.scene_id
                )
            )
