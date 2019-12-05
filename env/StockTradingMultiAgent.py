import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf

from env.StockTradingEnv import StockTradingEnv
from gym import spaces


class StockTradingMultiAgent(MultiAgentEnv):

    def __init__(self, num):
        super(StockTradingMultiAgent, self).__init__()

        self.agents = [StockTradingEnv() for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step_wrapper(action, 0.1)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

