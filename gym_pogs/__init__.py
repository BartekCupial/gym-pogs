from gym.envs.registration import register

from gym_pogs.envs.pogs import POGSEnv

register(
    id="POGS-v0",
    entry_point="gym_pogs:POGSEnv",
)
