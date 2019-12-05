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
        self.price = random.randint(1, 100)
        self.num = num
        self.initial_shares = np.zeros(num)


    def reset(self):
        #print('RESET ')
        self.resetted = True
        self.dones = set()
        self.price = random.randint(1,100)
        self.initial_shares = np.zeros(self.num)
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info, quantity = {}, {}, {}, {}, {}
        done["__all__"] = False
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step_wrapper(action, self.price)
            delta_shares = (obs[i][5][2] - self.initial_shares[i])*MAX_NUM_SHARES
            self.price = self.price_function(self.price, delta_shares)

            self.initial_shares[i] = obs[i][5][2]
            if done[i]:
               # print('Fatto')
                done["__all__"] = True

        return obs, rew, done, info

    def price_function(self, initial_price, delta_shares):
        #print('initial price: ', initial_price, 'second price: ', initial_price + (delta_shares )/initial_price, 'delta: ', delta_shares)

        return initial_price + (delta_shares)/initial_price
