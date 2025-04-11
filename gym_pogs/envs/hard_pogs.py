import gymnasium as gym

from gym_pogs.agents.memory_symbolic import MemorySymbolicPOGSAgent


class HardPOGS(gym.Wrapper):
    def __init__(self, env, min_backtracks: int = 3):
        super().__init__(env)

        self.min_backtracks = min_backtracks
        self.agent = MemorySymbolicPOGSAgent(self.env.k_nearest)

    def reset(self, *, seed=None, **kwargs):
        # Extract seed from kwargs if provided
        if seed is not None:
            # Set the random number generator with the provided seed
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        backtrack_count = 0

        while backtrack_count < self.min_backtracks:
            seed = int(self.np_random.integers(0, 2**32))

            self.agent.reset()
            obs, info = super().reset(seed=seed, **kwargs)

            done = False
            while not done:
                action = self.agent.act(obs)
                obs, reward, term, trun, info = self.env.step(action)
                done = term or trun

            backtrack_count = self.agent.backtrack_count

        self.agent.reset()
        return self.env.reset(seed=seed, **kwargs)
