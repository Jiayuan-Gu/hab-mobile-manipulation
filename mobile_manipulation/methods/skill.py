from collections import OrderedDict
from typing import Dict

from habitat import Config, RLEnv

from mobile_manipulation.common.registry import (
    mm_registry as my_registry,
)


class Skill:
    def __init__(self, config: Config, rl_env: RLEnv):
        self._config = config
        self._rl_env = rl_env
        self._obs_space = rl_env.observation_space
        self._action_space = rl_env.action_space

    def reset(self, obs, **kwargs):
        self._elapsed_steps = 0

    def act(self, obs, **kwargs) -> Dict:
        raise NotImplementedError

    def should_terminate(self, obs, **kwargs):
        raise NotImplementedError

    def is_timeout(self):
        timeout = self._config.get("TIMEOUT", 0)
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def to(self, device):
        return self


@my_registry.register_skill
class Wait(Skill):
    def act(self, obs, **kwargs):
        self._elapsed_steps += 1
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return self.is_timeout()


@my_registry.register_skill
class Terminate(Skill):
    def act(self, obs, **kwargs):
        self._rl_env.habitat_env.task._should_terminate = True
        return {"action": "EmptyAction"}

    def should_terminate(self, obs, **kwargs):
        return True


@my_registry.register_skill
class CompositeSkill(Skill):
    skills: Dict[str, Skill]
    _skill_idx: int
    _skill_reset: bool

    def __init__(
        self,
        config: Config,
        rl_env: RLEnv,
    ):
        self._config = config
        self._rl_env = rl_env

        self.skill_sequence = config.get("SKILL_SEQUENCE", config.SKILLS)
        self.skills = self._init_entities(
            entity_names=config.SKILLS,
            register_func=my_registry.get_skill,
            entities_config=config,
            rl_env=rl_env,
        )

    def _init_entities(
        self, entity_names, register_func, entities_config=None, **kwargs
    ) -> OrderedDict:
        """Modified from EmbodiedTask."""
        if entities_config is None:
            entities_config = self._config

        entities = OrderedDict()
        for entity_name in entity_names:
            entity_cfg = getattr(entities_config, entity_name)
            entity_type = register_func(entity_cfg.TYPE)
            assert (
                entity_type is not None
            ), f"invalid {entity_name} type {entity_cfg.TYPE}"
            entities[entity_name] = entity_type(config=entity_cfg, **kwargs)
        return entities

    @property
    def current_skill_name(self):
        return self.skill_sequence[self._skill_idx]

    @property
    def current_skill(self):
        if self._skill_idx >= len(self.skill_sequence):
            return None
        return self.skills[self.current_skill_name]

    def set_skill_idx(self, idx):
        if idx is None:
            self._skill_idx += 1
        else:
            self._skill_idx = idx

    def reset(self, obs, **kwargs):
        self.set_skill_idx(0)
        print("Skill <{}> begin.".format(self.current_skill_name))
        self.current_skill.reset(obs, **kwargs)

    def act(self, obs, **kwargs):
        if self.current_skill.should_terminate(obs, **kwargs):
            print("Skill <{}> terminate.".format(self.current_skill_name))
            self.set_skill_idx(None)
            if self.current_skill is not None:
                print("Skill <{}> begin.".format(self.current_skill_name))
                self.current_skill.reset(obs, **kwargs)

        if self.current_skill is not None:
            action = self.current_skill.act(obs, **kwargs)
            if action is None:  # nested composite skill terminate
                action = self.act(obs, **kwargs)
            return action

    def to(self, device):
        for skill in self.skills.values():
            skill.to(device)
        return self

    def should_terminate(self, obs, **kwargs):
        return self._skill_idx >= len(self.skill_sequence)
