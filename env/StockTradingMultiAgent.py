import random

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
import numpy as np
from env.StockTradingEnv import StockTradingEnv
from gym import spaces
import matplotlib.pyplot as plt

MAX_NUM_SHARES = 2147483647


class StockTradingMultiAgent(MultiAgentEnv):

    def __init__(self, num):
        super(StockTradingMultiAgent, self).__init__()
        self.agents = [StockTradingEnv() for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.price = random.randint(1, 10)
        self.num = num
        self.asks = np.empty((0, 5))
        self.bids = np.empty((0, 6))
        self.transaction = np.empty((0, 4))
        self.steppps = 0
    def reset(self):
        self.steppps = 0
        self.asks = np.empty((0, 5))
        self.bids = np.empty((0, 6))
        self.transaction = np.empty((0, 5))
        self.dones = set()
        self.price = random.randint(1, 10)
        self.prices = []
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info, quantity = {}, {}, {}, {}, {}
        done["__all__"] = False

        for i, action in action_dict.items():
            if np.isnan(action).any() == False:
                obs[i], rew[i], done[i], info[i], self.bids, self.asks, self.price, self.transaction = self.agents[i].step_wrapper(action, self.price, i, self.bids, self.asks, self.transaction, self.num)
                self.prices.append(self.price)
            # if (done[i] or self.steppps > 100):

        self.steppps += 1
        if self.steppps > 500:
            # print('Fuori ')
            plt.figure(figsize=(10, 5))
            plt.title('Price')
            plt.plot(range(0, len(self.prices)), self.prices)
            plt.show(block=False)
            plt.show()
            done["__all__"] = True

        return obs, rew, done, info