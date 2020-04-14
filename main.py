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
class Random(Policy):
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_space.seed(2)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        """No learning."""
        # return {}
        pass

    def get_weights( self):
        pass

    def set_weights(self, weights):
        pass


def select_policy(agent_id):
    if agent_id%2 ==0:
        return "pg_policy"
    else:
        return "pg_policy2"


   #return "pg_policy"


ray.init()

env = StockTradingMultiAgent(10)
register_env("test", lambda _: env)



trainer = ppo.PPOTrainer(env="test", config={
            "env":  "test",
            "num_gpus": 0,
            "num_workers": 7,
            "simple_optimizer": True,
            "multiagent": {
                "policies": {
                               "pg_policy": (None,  spaces.Box(
                                    low=0, high=1, shape=(10, 6), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                               "pg_policy2": (None,  spaces.Box(
                                    low=0, high=1, shape=(10, 6), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                              },
                "policy_mapping_fn": (lambda agent_id: select_policy(agent_id))   ,
                "policies_to_train": ["pg_policy",  "pg_policy2"],

            },
})

for i in range(2000):
   # Perform one iteration of training the policy with PPO
   print(i)
   result = trainer.train()
   print(pretty_print(result))
   if i % 10 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
