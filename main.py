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
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from env.StockTradingMultiAgent import StockTradingMultiAgent

def select_policy(agent_id):
        return "pg_policy"


ray.init()

env = StockTradingMultiAgent(10)
register_env("test", lambda _: env)



trainer = ppo.PPOTrainer(env="test",config={
            "env":  "test",
            "num_gpus": 0,
            "num_workers": 0,
            "simple_optimizer": True,
            "multiagent": {
                "policies": {
                               "pg_policy": (None,  spaces.Box(
                                    low=0, high=1, shape=(10, 5), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                               "pg_policy2": (None,  spaces.Box(
                                    low=0, high=1, shape=(10, 5), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                              },
                "policy_mapping_fn": (lambda agent_id: select_policy(2)),
                "policies_to_train": ["pg_policy",  "pg_policy2"],

            },

})
tune.run(
        "PPO",
        #name="PPO_discrete5",
        checkpoint_freq=10, # iterations
        checkpoint_at_end=True,
        restore= "~/ray_results/PPO/PPO_test_a2de55ee_2020-05-29_10-51-505z2op7c8/checkpoint_60/checkpoint-60",
        config={
            "env":  "test",
            "num_gpus": 0,
            "num_workers": 7,
            "simple_optimizer": True,
            "multiagent": {
                "policies": {
                               "pg_policy": (None,  spaces.Box(
                                    low=0, high=1, shape=(10, 5), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),

                              },
                "policy_mapping_fn": (lambda agent_id: select_policy(2)),
                "policies_to_train": ["pg_policy"],

            },

})
#for i in range(2000):
#   # Perform one iteration of training the policy with PPO
#   print(i)
#   result = trainer.train()
#   print(pretty_print(result))
#   if i % 5 == 0:
#       checkpoint = trainer.save()
#       print("checkpoint saved at", checkpoint)
