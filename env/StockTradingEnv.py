import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20
INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StockTradingEnv, self).__init__()
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, -1]), high=np.array([3, 0.2, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 6), dtype=np.float16)

        self.past_balance = 0
        self.past_net_worth = 0
    def _next_observation(self):

        # Append additional data and scale each value to between 0-1
        obs = np.array([[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]])

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        step_action_type = action[0]
        step_amount = action[1]
        step_percent_price = action[2]
        #print('current price')
        #print(self.current_price)
        #print('step_percent_price')
        #print(step_percent_price)
        step_price = self.current_price + self.current_price*step_percent_price/100
        #print('Balance')
        #print(self.balance)
        #print('step price')
        #print(step_price)
        step_total_possible = int(self.balance / step_price)
        step_bought_shares = int(step_amount * step_total_possible)
        find = True
       # print('fuori: ')
       # print(self.balance)
       # print('step action')
       # print(step_action_type)
        if step_action_type <= 1:
            # Buy step_amount % of balance in shares
           # print('Compro')
            if self.asks.size >= 0:
                j = 0
                for item in self.asks:
                    ask_shares_price = item[3]
                    ask_shares_bought = item[2]
                    ask_shares_bought_min = ask_shares_bought - (ask_shares_bought * 0.1)
                    ask_shares_bought_max = ask_shares_bought + (ask_shares_bought * 0.1)
                    ask_shares_cost = ask_shares_price * ask_shares_bought
                    if item[0] != self.i:
                        #print('Step price')
                        #print(step_price)
                        #print('ask_shares_price')
                        #print(ask_shares_price)
                        #print('ask_shares_bought')
                        #print(ask_shares_bought)
                        if step_price >= ask_shares_price and self.balance >= ask_shares_cost and step_bought_shares >= ask_shares_bought_min and step_bought_shares <= ask_shares_bought_max and find:
                            prev_cost = self.cost_basis * self.shares_held
                            #print('BALANCE')
                            #print(self.balance)
                            #print('Ask')
                            self.balance -= ask_shares_cost
                            #print('ASK')
                            #print('BALANCE')
                            #print(self.balance)
                            self.cost_basis = (prev_cost + ask_shares_cost) / (self.shares_held + ask_shares_bought)
                            self.current_price = ask_shares_price
                            #print('Price: ')
                            #print(self.current_price)
                            self.asks = np.delete(self.asks, j, axis=0)
                            self.transaction = np.append(self.transaction, [[self.i, step_action_type, ask_shares_bought,  ask_shares_price]], axis=0)
                            find = False
                            break
                    j = j + 1
        elif step_action_type >= 1 and step_action_type < 2:
            step_sold_shares = int(step_amount * self.shares_held)
            if self.bids.size > 0:
                j = 0
                for item in self.bids:
                    bids_shares_price = item[3]
                    bids_shares_sold = item[2]
                    bids_shares_sold_min = bids_shares_sold - (bids_shares_sold * 0.1)
                    bids_shares_sold_max = bids_shares_sold + (bids_shares_sold * 0.1)
                    if item[0] != self.i:
                        if step_price <= bids_shares_price and self.shares_held >= bids_shares_sold and step_sold_shares >= bids_shares_sold_min and step_sold_shares <= bids_shares_sold_max and find:
                           # print('BIDS: ')
                            self.shares_held -= bids_shares_sold
                          #  print('shares held')
                          #  print(self.shares_held#)
                            self.total_shares_sold += bids_shares_sold
                            self.total_sales_value += bids_shares_price
                            self.current_price = bids_shares_price
                            self.bids = np.delete(self.bids, j, 0)
                            self.transaction = np.append(self.transaction, [[self.i, step_action_type, bids_shares_sold, bids_shares_price]], axis=0)
                            find = False
                            break
                    j = j + 1
        if (find):
            if step_action_type <= 1 and self.balance >= step_price * int(step_total_possible * step_amount):
                self.bids = np.append(self.bids, [[self.i, step_action_type, step_bought_shares, step_price]], axis=0)
                self.balance -= step_price * int(step_total_possible * step_amount)
            elif step_action_type > 1 and step_action_type <= 2 and self.shares_held >= int(self.shares_held * step_amount):
                self.asks = np.append(self.asks, [[self.i, step_action_type, int(self.shares_held * step_amount), step_price]], axis=0)
                self.shares_held -= int(self.shares_held * step_amount)
        self.net_worth = self.balance + self.shares_held * self.current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        #print('STEEEEP')
        #print('current step')
        #print(self.current_step)
        #print('balance: ')
        #print(self.balance)
        self.current_step += 1
        if self.current_step > MAX_STEPS:
            self.current_step = 0
        delay_modifier = (self.current_step / MAX_STEPS)
        reward = np.exp(((self.balance - self.past_balance) / 1000)) * delay_modifier
        #print('Reward')
        #print(reward)
        #print('Balance: ')
        #print(self.balance)
        #print('shares')
        #print(self.shares_held)
        done = self.net_worth >= 10 * INITIAL_ACCOUNT_BALANCE
        obs = self._next_observation()
        self.past_balance = self.balance
        self.past_net_worth = self.net_worth
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        #print('REESETT')
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 100
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, MAX_STEPS)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
    def step_wrapper(self, action, price, i, bids, asks, transaction):
        self.bids = np.array(bids)
        self.i = i
       # print('Indice: ')
       # print(self.i)
        self.asks = np.array(asks)
        self.current_price = price
        self.transaction = transaction
        j = 0
        for item in self.transaction:
            transaction_agent_id = item[0]
            transaction_action_type = item[1]
            transaction_shares = item[2]
            transaction_price = item[3]
            if transaction_agent_id == self.i:
                if transaction_action_type < 1:
                    self.transaction = np.delete(self.transaction, [j], 0)
                    self.shares_held += transaction_shares
                elif transaction_action_type >= 1 and transaction_action_type < 2:
                    self.total_shares_sold += transaction_shares
                    self.balance += transaction_price * transaction_shares
                    self.transaction = np.delete(self.transaction, [j], 0)
            j += 1
        obs, rew, done, info = self.step(action)
       # print('Agente: ')
       # print(self.i)
       # print('Net worth')
       # print(self.net_worth)
       # print('price')
       # print(self.current_price)
        return obs, rew, done, info, self.bids, self.asks, self.current_price, self.transaction