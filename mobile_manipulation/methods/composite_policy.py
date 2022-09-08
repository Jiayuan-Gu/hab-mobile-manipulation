from collections import OrderedDict
from typing import Dict

from gym import spaces
from habitat import Config, RLEnv

from mobile_manipulation.common.registry import (
    mm_registry as my_registry,
)
from mobile_manipulation.methods.skill import Skill


class CompositePolicy:
    """Policy composed of a sequence of skills."""

    skills: Dict[str, Skill]

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

    def reset(self, obs, **kwargs):
        for _, skill in self.skills.items():
            skill.reset(obs, **kwargs)
        self._curr_skill_idx = 0
        self._curr_skill_name = self.skill_sequence[0]

    def act(self, obs, **kwargs):
        curr_skill = self.skills[self._curr_skill_name]

        if curr_skill.should_terminate(obs, **kwargs):
            print("Skill <{}> terminate.".format(self._curr_skill_name))

            self._curr_skill_idx += 1
            self._curr_skill_name = self.skill_sequence[self._curr_skill_idx]
            print("Skill <{}> begin.".format(self._curr_skill_name))

            curr_skill = self.skills[self._curr_skill_name]
            curr_skill.reset(obs, **kwargs)

        action = curr_skill.act(obs, **kwargs)

        # -------------------------------------------------------------------------- #
        # DEBUG: check value function
        # -------------------------------------------------------------------------- #
        # action = {}
        # values = {}
        # for skill_name, skill in self.skills.items():
        #     if "RL" not in skill_name:
        #         continue
        #     a = skill.act(obs, **kwargs)
        #     # print(skill_name, a["value"])
        #     if skill_name == self._curr_skill_name:
        #         action = a
        #     values[skill_name] = a["value"]
        # if len(action) == 0:
        #     action = curr_skill.act(obs, **kwargs)
        # action["values"] = values
        # -------------------------------------------------------------------------- #

        return action

    def to(self, device):
        for skill in self.skills.values():
            skill.to(device)
        return self
