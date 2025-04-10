from gym.envs.registration import register

from gym_pogs.envs.pogs import MinDistancePOGS, POGSEnv

__all__ = [POGSEnv, MinDistancePOGS]

register(
    id="POGS-v0",
    entry_point="gym_pogs:POGSEnv",
)
register(
    id="MinDistancePOGS-v0",
    entry_point="gym_pogs:MinDistancePOGS",
)
