from typing import Optional

from habitat.core.registry import Registry


class MobileManipulationRegistry(Registry):
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.ppo.policy import ActorCritic

        return cls._register_impl(
            "policy", to_register, name, assert_type=ActorCritic
        )

    @classmethod
    def get_policy(cls, name: str):
        return cls._get_impl("policy", name)

    @classmethod
    def register_skill(cls, to_register=None, *, name: Optional[str] = None):
        # NOTE(jigu): import on-the-fly to avoid import loop
        from mobile_manipulation.methods.skill import Skill

        return cls._register_impl(
            "skill", to_register, name, assert_type=Skill
        )

    @classmethod
    def get_skill(cls, name: str):
        return cls._get_impl("skill", name)


mm_registry = MobileManipulationRegistry()
