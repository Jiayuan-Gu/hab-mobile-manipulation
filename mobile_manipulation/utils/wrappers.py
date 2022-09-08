from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from habitat import RLEnv, logger


def flatten_action_spaces(action_spaces: spaces.Dict):
    action_shape = []
    low = []
    high = []
    action_mapping = OrderedDict()
    action_offset = 0

    for action_name, action_space in action_spaces.spaces.items():
        assert (
            isinstance(action_space, spaces.Box)
            and len(action_space.shape) == 1
        ), (
            action_name,
            action_space,
        )

        action_dim = action_space.shape[0]
        action_shape.append(action_dim)
        low.append(action_space.low)
        high.append(action_space.high)
        action_mapping[action_name] = (
            action_offset,
            action_offset + action_dim,
        )
        action_offset += action_dim

    action_shape = sum(action_shape)
    return (
        spaces.Box(
            shape=(action_shape,),
            low=np.hstack(low),
            high=np.hstack(high),
            dtype=np.float32,
        ),
        action_mapping,
    )


def flatten_observation_spaces(observation_spaces: spaces.Dict):
    obs_shape = []
    low = []
    high = []
    for obs_name, obs_space in observation_spaces.spaces.items():
        assert (
            isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1
        ), (obs_name, obs_space)
        obs_shape.append(obs_space.shape[0])
        low.append(obs_space.low)
        high.append(obs_space.high)
    obs_shape = sum(obs_shape)
    return spaces.Box(
        shape=(obs_shape,),
        low=np.hstack(low),
        high=np.hstack(high),
        dtype=np.float32,
    )


class HabitatActionWrapper(gym.ActionWrapper):
    """A (action) gym wrapper for Habitat RLEnv.
    Use a single and flatten action, according to RL config.
    """

    env: RLEnv

    def __init__(self, env: RLEnv) -> None:
        super().__init__(env)

        self._action_name = env._rl_config.ACTION_NAME
        self._task_action = env.habitat_env.task.actions[self._action_name]
        self._action_space = self.action_space[self._action_name]
        if isinstance(self._action_space, spaces.Dict):
            self.action_space, self.action_mapping = flatten_action_spaces(
                self._action_space
            )
        else:
            self.action_space = self._action_space
            self.action_mapping = {"action": (0, None)}
        # logger.info(self.action_mapping)

    def action(self, action):
        if isinstance(action, np.ndarray):
            # If task action has implemented its own way
            if hasattr(self._task_action, "get_action_args"):
                action_args = self._task_action.get_action_args(action)
                step_action = {
                    "action": self._action_name,
                    "action_args": action_args,
                }
            else:
                action_args = {
                    k: action[start:end]
                    for k, (start, end) in self.action_mapping.items()
                }
                step_action = {
                    "action": self._action_name,
                    "action_args": action_args,
                }
        elif isinstance(action, dict):
            step_action = action.copy()
            assert "action_args" in step_action, step_action
            if "action" not in step_action:
                step_action["action"] = self._action_name
        elif isinstance(action, int):
            step_action = {
                "action": self._action_name,
                "action_args": {"action": action},
            }
        else:
            raise NotImplementedError(action)
        return step_action

    @property
    def task_action(self):
        return self.env.habitat_env.task.actions[self._action_name]


class HabitatActionWrapperV1(gym.ActionWrapper):
    env: RLEnv

    def __init__(self, env: RLEnv) -> None:
        super().__init__(env)

        assert isinstance(self.action_space, spaces.Dict), self.action_space
        new_action_space = {}
        new_action_mapping = {}
        for name, action_space in self.action_space.spaces.items():
            if isinstance(action_space, spaces.Dict):
                (
                    new_action_space[name],
                    new_action_mapping[name],
                ) = flatten_action_spaces(action_space)
            else:
                new_action_space[name] = action_space
                new_action_mapping[name] = {"action": (0, None)}

        self.action_space = spaces.Dict(new_action_space)
        self.action_mapping = new_action_mapping
        logger.info(self.action_mapping)

    def action(self, action_dict):
        action_dict = action_dict.copy()
        assert isinstance(action_dict, dict), action_dict

        action_name = action_dict["action"]

        if "action_args" in action_dict:
            action_args = action_dict["action_args"]
            if isinstance(action_args, np.ndarray):
                task_action = self.env._env.task.actions[action_name]
                if hasattr(task_action, "get_action_args"):
                    action_args = task_action.get_action_args(action_args)
                else:
                    action_args = {
                        k: action_args[start:end]
                        for k, (start, end) in self.action_mapping[
                            action_name
                        ].items()
                    }
                action_dict["action_args"] = action_args

        return action_dict
