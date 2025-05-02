from gymnasium.envs.registration import register

from gym_pogs.envs.expert_info import ExpertInfo
from gym_pogs.envs.hard_pogs import HardPOGS
from gym_pogs.envs.pogs import POGSEnv

__all__ = [POGSEnv, HardPOGS]


def make_pogs(expert_penalty=-1, **kwargs):
    env = POGSEnv(**kwargs)
    env = ExpertInfo(env)

    return env


def make_hard_pogs(expert_penalty=-1, min_backtracks=3, **kwargs):
    env = POGSEnv(**kwargs)
    env = HardPOGS(env, min_backtracks=min_backtracks)
    env = ExpertInfo(env, expert_penalty=expert_penalty)

    return env


register(
    id="POGS-v0",
    entry_point=make_pogs,
    max_episode_steps=None,
    kwargs={},
)

register(
    id="HardPOGS-v0",
    entry_point=make_hard_pogs,
    max_episode_steps=None,
    kwargs={},
)
