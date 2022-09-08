import habitat_sim
import magnum as mn
import numpy as np
from habitat.core.env import Env

# from habitat_extensions.tasks.tidy_house.sim import TidyHouseSim
# from habitat_extensions.tasks.tidy_house.task import TidyHouseTask
from habitat_extensions.tasks.rearrange.sim import RearrangeSim
from habitat_extensions.tasks.rearrange.task import RearrangeTask
from habitat_extensions.utils.geo_utils import wrap_angle
from mobile_manipulation.common.registry import (
    mm_registry as my_registry,
)

from ..skill import Skill


@my_registry.register_skill
class NavGTSkill(Skill):
    def reset(self, obs, **kwargs):
        self._sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = self._sim.robot
        self._task: RearrangeTask = self._rl_env.habitat_env.task
        self._goal = self._task.goal_to_nav

        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(
            self._sim.robot.base_pos, dtype=np.float32
        )
        path.requested_end = np.array(self._goal[0:3], dtype=np.float32)
        found_path = self._sim.pathfinder.find_path(path)
        assert found_path, "No path is found for episode {}".format(
            self._rl_env.habitat_env.current_episode.episode_id
        )
        self.path = path
        self.path_length = len(path.points)
        print("Path length is {}".format(self.path_length))
        self._path_index = 1

        self._elapsed_steps = 0

    def _compute_velocity(
        self, goal_pos, goal_ori, dist_thresh, ang_thresh, turn_thresh
    ):
        base_invT = self._robot.base_T.inverted()
        base_pos = self._robot.base_pos

        direction_world = goal_pos - base_pos[[0, 2]]
        direction_base = base_invT.transform_vector(
            mn.Vector3(direction_world[0], 0, direction_world[1])
        )
        direction = np.array(direction_base)

        distance = np.linalg.norm(direction)
        should_stop = False

        if distance < dist_thresh:
            lin_vel = 0.0

            if goal_ori is None:
                angle = np.arctan2(-direction[2], direction[0])
            else:
                angle = wrap_angle(goal_ori - self._robot.base_angle)
            if ang_thresh is None or np.abs(angle) <= np.deg2rad(ang_thresh):
                ang_vel = 0.0
                should_stop = True
            else:
                ang_vel = angle / self._sim.timestep
        else:
            angle = np.arctan2(-direction[2], direction[0])
            if np.abs(angle) <= np.deg2rad(turn_thresh):
                lin_vel = distance * np.cos(angle) / self._sim.timestep
            else:
                lin_vel = 0.0
            ang_vel = angle / self._sim.timestep

        # print(lin_vel, ang_vel)
        # print(distance, np.rad2deg(angle))

        return lin_vel, ang_vel, should_stop

    def act(self, obs, **kwargs):
        # # Kinematically set the robot
        # self._sim.robot.base_pos = self.path.points[self._elapsed_steps]
        # self._sim.robot.base_angle = self._goal[-1]
        # self._elapsed_steps += 1
        # return {"action": "EmptyAction"}

        goal_pos = self.path.points[self._path_index][[0, 2]]
        if self._path_index == self.path_length - 1:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                self._goal[-1],
                dist_thresh=self._config.DIST_THRESH,
                ang_thresh=self._config.ANG_THRESH,
                turn_thresh=self._config.TURN_THRESH,
            )
            if should_stop:
                self._path_index += 1
                print("Finish the last waypoint", self._path_index)
        else:
            lin_vel, ang_vel, should_stop = self._compute_velocity(
                goal_pos,
                None,
                # dist_thresh=self._config.DIST_THRESH,
                dist_thresh=0.05,
                ang_thresh=None,
                turn_thresh=self._config.TURN_THRESH,
            )
            if should_stop:
                self._path_index += 1
                print("Advanced to next waypoint", self._path_index)
        step_action = {
            "action": "BaseVelAction",
            "action_args": {"velocity": [lin_vel, ang_vel]},
        }

        self._elapsed_steps += 1
        return step_action

    def is_timeout(self):
        timeout = self._config.get("TIMEOUT", 0)
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def should_terminate(self, obs, **kwargs):
        if self._path_index == self.path_length:
            return True
        return self.is_timeout()


@my_registry.register_skill
class ResetArm(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        sim: RearrangeSim = self._rl_env.habitat_env.sim
        self._robot = sim.robot
        self.ee_tgt_pos = np.array([0.5, 0.0, 1.0])
        self._set_ee_tgt_pos()

        sim.pyb_robot.set_joint_states(self._robot.params.arm_init_params)
        arm_tgt_qpos = sim.pyb_robot.IK(self.ee_tgt_pos, max_iters=100)
        cur_qpos = np.array(self._robot.arm_joint_pos)
        tgt_qpos = np.array(arm_tgt_qpos)
        n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.1)
        n_step = max(1, int(n_step))
        self.plan = np.linspace(cur_qpos, tgt_qpos, n_step)
        self._plan_idx = 0

    def _set_ee_tgt_pos(self):
        # task = self._rl_env.habitat_env.task
        task_actions = self._rl_env.habitat_env.task.actions
        action_names = ["ArmGripperAction", "BaseArmGripperAction"]
        for action_name in action_names:
            if action_name not in task_actions:
                continue
            task_actions[action_name].ee_tgt_pos = self.ee_tgt_pos

    def act(self, obs, **kwargs):
        self._robot.arm_motor_pos = self.plan[self._plan_idx]
        self._plan_idx += 1
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        if self.is_timeout():
            return True
        return self._plan_idx >= len(self.plan)


@my_registry.register_skill
class NextTarget(Skill):
    def reset(self, obs, **kwargs):
        super().reset(obs, **kwargs)
        task = self._rl_env.habitat_env.task
        task.set_target(task.tgt_idx + 1)

    def act(self, obs, **kwargs):
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return True
