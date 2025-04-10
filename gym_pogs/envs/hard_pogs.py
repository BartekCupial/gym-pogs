import gym

from gym_pogs.agents.memory_symbolic import MemorySymbolicPOGSAgent


class HardPOGS(gym.Wrapper):
    def __init__(self, env, min_backtracks: int = 3):
        super().__init__(env)

        self.min_backtracks = min_backtracks
        self.agent = MemorySymbolicPOGSAgent()

    def reset(self, **kwargs):
        backtrack_count = 0

        while backtrack_count < self.min_backtracks:
            seed = self.np_random.randint(0, 2**32)

            self.env.seed(seed)
            obs = super().reset(**kwargs)
            self.agent.reset(obs)

            done = False
            while not done:
                action = self.agent.act(obs)
                obs, reward, done, info = super().step(action)

            backtrack_count = self.agent.backtrack_count

        self.env.seed(seed)
        return super().reset(**kwargs)
