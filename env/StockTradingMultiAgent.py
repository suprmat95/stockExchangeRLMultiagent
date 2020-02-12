import random

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
import numpy as np
from env.StockTradingEnv import StockTradingEnv
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd


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
        columns = list(map(lambda n: f"Agent {n}", range(1, self.num)))
        self.df_net_worthes = pd.DataFrame()

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
        obs, rew, done, info, quantity, net_worthes = {}, {}, {}, {}, {}, {}
        done["__all__"] = False

        for i, action in action_dict.items():
            if np.isnan(action).any() == False:
                obs[i], rew[i], done[i], info[i], net_worthes[i], self.bids, self.asks, self.price, self.transaction = self.agents[i].step_wrapper(action, self.price, i, self.bids, self.asks, self.transaction, self.num)
                self.prices.append(self.price)
            # if (done[i] or self.steppps > 100):

        self.steppps += 1
        if self.steppps > 500:
            #map(lambda n: n,)
            print('Fuori ')
            self.df_net_worthes = pd.DataFrame(net_worthes)
            #print(pd.DataFrame(net_worthes))
            plt.figure(figsize=(10, 5))
            plt.title('Price')
            plt.plot(range(0, len(self.prices)), self.prices)
            plt.show(block=False)
            plt.show()
            plt.figure(figsize=(10, 5))
            x = 0
            y = 1
            plt.title('Net_Worth Comparision')
            plt.xlabel(f"Agent {x}")
            plt.ylabel(f"Agent {y}")
            colors = range(0, self.df_net_worthes[0].size)
            plt.scatter(self.df_net_worthes[x], self.df_net_worthes[y], c=colors, cmap='Greens')
            cbar = plt.colorbar()
            cbar.set_label('Steps')
            plt.show(block=False)
            plt.show()
            done["__all__"] = True

        return obs, rew, done, info

