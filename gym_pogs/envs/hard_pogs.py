import gym

from gym_pogs.agents.memory_symbolic import MemorySymbolicPOGSAgent


class HardPOGS(gym.Wrapper):
    def __init__(self, env, min_explore_paths: int = 3):
        super().__init__(env)

        self.min_explore_paths = min_explore_paths
        self.agent = MemorySymbolicPOGSAgent()

    def reset(self, **kwargs):
        num_explore_paths = 0

        while num_explore_paths < self.min_explore_paths:
            seed = self.np_random.randint(0, 2**32)

            self.env.seed(seed)
            obs = super().reset(**kwargs)
            self.agent.reset(obs)

            done = False
            while not done:
                action = self.agent.act(obs)
                obs, reward, done, info = super().step(action)

            num_explore_paths = self.agent.num_explore_paths

        self.env.seed(seed)
        return super().reset(**kwargs)
