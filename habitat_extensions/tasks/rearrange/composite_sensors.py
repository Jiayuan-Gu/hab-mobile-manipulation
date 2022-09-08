import re
from collections import OrderedDict

import magnum as mn
import numpy as np
from gym import spaces
from habitat import Config
from habitat.core.registry import registry

from habitat_extensions.robots.marker import Marker

from .sensors import MyMeasure
from .task import RearrangeTask


class Predicate:
    def __init__(self, task: RearrangeTask, predicate: str) -> None:
        self._task = task
        self._sim = task._sim
        self.parse_predicate(predicate)

    def parse_predicate(self, predicate: str):
        self._predicate = predicate
        m = re.match(r"(?P<predicate>.+)\((?P<args>.*)\)", predicate)
        if m is None:
            raise ValueError(predicate)
        mapping = m.groupdict()

        predicate_name = mapping["predicate"]
        self._predicate_fn = getattr(self, f"predicate_{predicate_name}")

        args = mapping["args"].split(",")
        self._predicate_args = [self.parse_arg(arg) for arg in args]

    def parse_arg(self, arg: str):
        args = arg.split("|")
        bind_fn = getattr(self, "bind_{}".format(args[0] or "none"))
        return bind_fn, args[1:]

    def is_satisfied(self):
        predicate_args = [
            bind_fn(*bind_args) for bind_fn, bind_args in self._predicate_args
        ]
        return self._predicate_fn(*predicate_args)

    def predicate_at(self, pos1, pos2, threshold=0.15):
        return np.linalg.norm(pos1 - pos2) <= threshold

    def predicate_holding(self, obj):
        return self._sim.gripper.grasped_obj == obj

    def predicate_not_holding(self, *args):
        return not self._sim.gripper.is_grasped

    def predicate_opened_drawer(self, marker: Marker):
        return marker.qpos >= 0.4

    def predicate_closed_drawer(self, marker: Marker):
        return marker.qpos <= 0.1

    def predicate_opened_fridge(self, marker: Marker):
        return marker.qpos >= 1.57 - 0.15

    def predicate_closed_fridge(self, marker: Marker):
        return marker.qpos <= 0.15

    def predicate_robot_at(self, pos, thresh):
        offset = (self._sim.robot.base_pos - pos)[[0, 2]]
        dist = np.linalg.norm(offset)
        return dist <= thresh

    def predicate_ee_at_rest(self, thresh: float):
        ee_pos_at_base = self._sim.robot.transform(
            self._sim.robot.gripper_pos, "world2base"
        )
        dist = np.linalg.norm(ee_pos_at_base - self._task.resting_position)
        return dist <= thresh

    def bind_target_obj(self, index=None):
        if index is None:
            index = self._task.tgt_idx
        return self._sim.get_target(int(index))[0]

    def bind_target_obj_pos(self, index=None):
        if index is None:
            index = self._task.tgt_idx
        tgt_obj = self._sim.get_target(int(index))[0]
        return np.array(tgt_obj.translation, dtype=np.float32)

    def bind_target_start_pos(self, index=None):
        if index is None:
            index = self._task.tgt_idx
        tgt_obj = self._sim.get_target(int(index))[0]
        return self._task.obj_start_pos[tgt_obj.handle]

    def bind_target_goal_pos(self, index=None):
        if index is None:
            index = self._task.tgt_idx
        tgt_T = self._sim.get_target(int(index))[1]
        return np.array(tgt_T.translation, dtype=np.float32)

    def bind_target_marker(self, index=None):
        if index is None:
            index = self._task.tgt_idx
        receptacle_handle, receptacle_link_id = self._task._target_receptacles[
            int(index)
        ]
        if "frige" in receptacle_handle or "fridge" in receptacle_handle:
            return self._sim.markers["fridge_push_point"]
        elif (
            "kitchen_counter" in receptacle_handle and receptacle_link_id != 0
        ):
            marker_name = "cab_push_point_{}".format(receptacle_link_id)
            return self._sim.markers[marker_name]
        else:
            raise NotImplementedError(receptacle_handle)

    def bind_none(self, *args):
        return None

    def bind_float(self, value):
        return float(value)


@registry.register_measure
class StageSuccess(MyMeasure):
    cls_uuid = "stage_success"

    def __init__(
        self,
        *args,
        task: RearrangeTask,
        config: Config,
        **kwargs,
    ):
        super().__init__(*args, config=config, task=task, **kwargs)
        self.goals = OrderedDict(
            (name, [Predicate(task, p) for p in predicates])
            for name, predicates in config.GOALS.items()
        )

    def reset_metric(self, *args, **kwargs):
        self._metric = {name: False for name in self.goals}

    def update_metric(self, *args, **kwargs):
        # self._metric = {
        #     name: all(p.is_satisfied() for p in goal)
        #     if not self._metric[name]
        #     else True
        #     for name, goal in self.goals.items()
        # }

        for name, goal in self.goals.items():
            if self._metric[name]:
                continue
            else:
                self._metric[name] = all(p.is_satisfied() for p in goal)
            if not self._metric[name]:
                break


# -------------------------------------------------------------------------- #
# For monolithic RL
# -------------------------------------------------------------------------- #
@registry.register_measure
class CompositeSuccess(MyMeasure):
    cls_uuid = "composite_success"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [StageSuccess.cls_uuid]
        )
        self._measure = task.measurements.measures[StageSuccess.cls_uuid]
        self._goal = list(self._measure.goals.keys())[-1]
        return super().reset_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, **kwargs) -> None:
        self._metric = self._measure.get_metric()[self._goal]


@registry.register_measure
class CompositeReward(MyMeasure):
    cls_uuid = "composite_reward"

    def __init__(self, *args, task: RearrangeTask, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._measures = task._init_entities(
            self._config.MEASUREMENTS, registry.get_measure, self._config
        )
        self._stage_measures = []
        for stage_measure in self._config.STAGE_MEASURES:
            self._stage_measures.append(
                [self._measures[x] for x in stage_measure]
            )

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        # In fact, this reward should be placed at last
        deps = [StageSuccess.cls_uuid]
        if self._config.get("RESET_GS", False):
            measures = task.measurements.measures
            if "gripper_status" in measures:
                deps.append("gripper_status")
            if "gripper_status_v1" in measures:
                deps.append("gripper_status_v1")
        task.measurements.check_measure_dependencies(self.uuid, deps)
        stage_successes = task.measurements.measures[StageSuccess.cls_uuid]
        assert len(stage_successes.goals) + 1 == len(self._stage_measures)

        self._stage_idx = -1
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        stage_successes = measures[StageSuccess.cls_uuid].get_metric()
        reward = 0.0

        # Infer the current stage
        stage_idx = 0
        for succ in stage_successes.values():
            if not succ:
                break
            stage_idx += 1

        # stage_measure = self._stage_measures[stage_idx]
        stage_measures = self._stage_measures[stage_idx]

        if self._stage_idx != stage_idx:
            assert self._stage_idx < stage_idx, (self._stage_idx, stage_idx)
            self._stage_idx = stage_idx

            # Reset gripper status
            if self._config.get("RESET_GS", False):
                self._reset_gs(*args, task=task, **kwargs)

            # Reset measures
            for stage_measure in stage_measures:
                stage_measure.reset_metric(
                    *args, task=task, no_dep=True, **kwargs
                )

            # Reward for stage success
            if stage_idx > 0:
                reward += self._config.STAGE_REWARD

        for stage_measure in stage_measures:
            stage_measure.update_metric(*args, task=task, **kwargs)
            reward += stage_measure.get_metric()

        self._metric = reward

    def _reset_gs(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        if "gripper_status" in measures:
            gs = measures["gripper_status"]
            gs.reset_metric(*args, task=task, **kwargs)
        if "gripper_status_v1" in measures:
            gs = measures["gripper_status_v1"]
            gs.reset_metric(*args, task=task, **kwargs)
