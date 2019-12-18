import random

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
import numpy as np
from env.StockTradingEnv import StockTradingEnv
from gym import spaces
MAX_NUM_SHARES = 2147483647


class StockTradingMultiAgent(MultiAgentEnv):

    def __init__(self, num):
        super(StockTradingMultiAgent, self).__init__()

        self.agents = [StockTradingEnv() for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False
        print('init')
        self.price = random.randint(1, 10)
        self.num = num
        self.asks = np.empty((0,4))
        self.bids = np.empty((0,4))
        self.transaction = np.zeros((1,2))


#

    def reset(self):
        #print('RESET ')
        self.resetted = True
        self.dones = set()
        print("STEP ¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶")
        self.price = random.randint(1, 10)
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info, quantity = {}, {}, {}, {}, {}
        done["__all__"] = False
        for i, action in action_dict.items():
            print('prima')
            print(self.price)
            if (np.isnan(action).any()):
                print('è nan')
            else:
                obs[i], rew[i], done[i], info[i], self.bids, self.asks, self.price, self.transaction = self.agents[i].step_wrapper(action, self.price, i, self.bids, self.asks, self.transaction)
                print('dopo')
                print(self.price)
                if done[i]:
                   # print('Fatto')
                    done["__all__"] = True
        return obs, rew, done, info