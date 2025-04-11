import gymnasium as gym
from gymnasium.envs.registration import register

from gym_pogs.envs.hard_pogs import HardPOGS
from gym_pogs.envs.pogs import POGSEnv

__all__ = [POGSEnv, HardPOGS]


def make_hard_pogs(min_backtracks=3, **kwargs):
    env = POGSEnv(**kwargs)
    return HardPOGS(env, min_backtracks=min_backtracks)


register(
    id="HardPOGS-v0",
    entry_point=make_hard_pogs,
    max_episode_steps=50,
    kwargs={},
)

register(
    id="POGS-v0",
    entry_point="gym_pogs:POGSEnv",
    max_episode_steps=50,
    kwargs={},
)
