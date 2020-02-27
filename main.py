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
import ray.rllib.agents.ppo as ppo


from env.StockTradingMultiAgent import StockTradingMultiAgent
from env.StockTradingEnv import StockTradingEnv
from ray.rllib.tests.test_multi_agent_env import MultiCartpole


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
           return "pg_policy"

   #return "pg_policy"


#asks = np.empty((0,4))
#print(asks)
#asks = np.append(asks, [[0.99,0.45,0.2,0.12]], axis=0)
#print(asks)
    #asks = np.delete(asks, [1], axis=0)
#print(asks)
ray.init()


register_env("test", lambda _: StockTradingMultiAgent(15))
trainer = ppo.PPOTrainer(env="test")
#        stop={"training_iteration": 5},

analysis = tune.run(
        "PPO",
        checkpoint_at_end=True,
        config={
            "env":  "test",
            "num_gpus": 0,
            "num_workers": 0,
            "simple_optimizer": True,
            "multiagent": {
                "policies": {
                                "pg_policy": (None,  spaces.Box(
                                    low=0, high=1, shape=(1, 7), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                                "random": (None,  spaces.Box(
                                    low=0, high=1, shape=(1, 7), dtype=np.float16),  spaces.Box(
                                    low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16), {}),
                              },
                "policy_mapping_fn": (lambda agent_id: select_policy(agent_id)),
                "policies_to_train": ["pg_policy"],

            },
        },
        )


# Get a dataframe for the max accuracy seen for each trial
df = analysis.dataframe()

# Get a dict mapping {trial logdir -> dataframes} for all trials in the experiment.
all_dataframes = analysis.trial_dataframes

# Get a list of trials
trials = analysis.trials
