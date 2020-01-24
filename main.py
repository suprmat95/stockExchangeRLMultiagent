import random

import gym
import json
import datetime as dt
import ray
from gym import spaces
from gym.spaces import Discrete
from ray import tune
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.tune import register_env

from env.StockTradingMultiAgent import StockTradingMultiAgent
from env.StockTradingEnv import StockTradingEnv
from ray.rllib.tests.test_multi_agent_env import MultiCartpole




def select_policy(agent_id):
       if agent_id == "player1":
           return "default_policy"
       else:
           return random.choice(["default_policy", "default_policy"])


#asks = np.empty((0,4))
#print(asks)
#asks = np.append(asks, [[0.99,0.45,0.2,0.12]], axis=0)
#print(asks)
#asks = np.delete(asks, [1], axis=0)
#print(asks)
ray.init()


register_env("test", lambda _: StockTradingMultiAgent(2))

tune.run(
        "PPO",
        stop={"episode_reward_mean": 10},
        config={
            "env":  "test",
            "num_gpus": 0,
            "num_workers": 0,
            "eager_tracing": False,
            "eager": False,
            "simple_optimizer": True,
            "multiagent": {
                "policies": {
                                "pg_policy": (None,  spaces.Box(
                                    low=0, high=1, shape=(1, 6), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0, -1]), high=np.array([3, 2, 10]), dtype=np.float16), {}),
                                "random": (None,  spaces.Box(
                                    low=0, high=1, shape=(1, 6), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0, -1]), high=np.array([3, 2, 1]), dtype=np.float16), {}),
                              },
                "policy_mapping_fn": (lambda agent_id: ["pg_policy", "random"][agent_id % 2]),
                "policies_to_train": ["pg_policy", "random"],

            },
        })
