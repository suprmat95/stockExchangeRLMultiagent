import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 500
INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StockTradingEnv, self).__init__()

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0, 0.01, -1]), high=np.array([3, 0.2, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 6), dtype=np.float16)
        self.max_balance = INITIAL_ACCOUNT_BALANCE
        self.num = 0
        self.i = 0
        self.balances = []
        self.shares_held_array = []

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
        step_price = self.current_price + self.current_price*step_percent_price/100
        step_total_possible = int(self.virtual_balance / step_price)
        step_bought_shares = int(step_amount * step_total_possible)
        find = True
        if step_action_type < 1:
            # Buy step_amount % of balance in shares
           # print('Compro')
            if self.asks.size > 0:
                j = 0
                for item in self.asks:
                    ask_agent = item[0]
                    ask_shares_price = item[3]
                    ask_shares_bought = item[2]
                    ask_step = item[4]
                    ask_shares_bought_min = ask_shares_bought - (ask_shares_bought * 0.1)
                    ask_shares_bought_max = ask_shares_bought + (ask_shares_bought * 0.1)
                    ask_shares_cost = ask_shares_price * ask_shares_bought
                    if ask_agent != self.i:
                        if ask_step >= (self.current_step - (self.num*10)) and step_price >= ask_shares_price and self.virtual_balance > ask_shares_cost and step_bought_shares >= ask_shares_bought_min and step_bought_shares <= ask_shares_bought_max and find:
                            prev_cost = self.cost_basis * self.shares_held
                            self.balance -= ask_shares_cost
                            self.virtual_balance -= ask_shares_cost
                            self.shares_held += ask_shares_bought
                            self.virtual_shares_held += ask_shares_bought
                            self.cost_basis = (prev_cost + ask_shares_cost) / (self.shares_held + ask_shares_bought)
                            self.current_price = ask_shares_price
                            self.ts = np.append(self.ts, [[step_action_type, ask_agent, ask_shares_bought, ask_shares_price]],    axis=0)
                            self.asks = np.delete(self.asks, [j], axis=0)
                            self.transaction = np.append(self.transaction, [[ask_agent, item[1], ask_shares_bought,  ask_shares_price, self.i]], axis=0)
                            find = False
                            break
                    j = j + 1
        elif step_action_type >= 1 and step_action_type < 2:
            #VENDO
            step_sold_shares = int(step_amount * self.shares_held)
            if self.bids.size > 0:
                j = 0
                for item in self.bids:
                    bids_agent = item[0]
                    bids_shares_price = item[3]
                    bids_step = item[4]
                    bids_shares_sold = item[2]
                    bids_shares_sold_min = bids_shares_sold - (bids_shares_sold * 0.1)
                    bids_shares_sold_max = bids_shares_sold + (bids_shares_sold * 0.1)
                    if bids_agent != self.i:
                        if bids_step >= (self.current_step - (self.num*10)) and step_price <= bids_shares_price and self.virtual_shares_held > bids_shares_sold and step_sold_shares >= bids_shares_sold_min and step_sold_shares <= bids_shares_sold_max and find:
                           # print('BIDS: ')
                           # print(f'Vendo Shares held: {self.shares_held}')
                           # print(f'Vendo Virtual Shares held: {self.virtual_shares_held}')
                           # print(f'Vendo Shares: {bids_shares_sold}')
                            self.shares_held -= bids_shares_sold
                            self.virtual_shares_held -= bids_shares_sold
                            self.balance += bids_shares_price * bids_shares_sold
                            self.virtual_balance += bids_shares_price * bids_shares_sold
                            self.total_shares_sold += bids_shares_sold
                            self.total_sales_value += bids_shares_price
                            self.current_price = bids_shares_price
                            self.ts = np.append(self.ts, [[step_action_type, bids_agent, bids_shares_sold, bids_shares_price]],    axis=0)
                            self.bids = np.delete(self.bids, [j], 0)
                            self.transaction = np.append(self.transaction, [[bids_agent, item[1], bids_shares_sold, bids_shares_price, self.i]], axis=0)
                            find = False
                            break
                    j = j + 1
        if (find):
            if step_action_type < 1 and self.virtual_balance >= step_price * int(step_total_possible * step_amount):
                self.bids = np.append(self.bids, [[self.i, step_action_type, step_bought_shares, step_price, self.current_step]], axis=0)
                self.virtual_balance -= step_price * int(step_total_possible * step_amount)
            elif step_action_type >= 1 and step_action_type < 2 and self.virtual_shares_held >= int(self.virtual_shares_held * step_amount):
                self.asks = np.append(self.asks, [[self.i, step_action_type, int(self.virtual_shares_held * step_amount), step_price, self.current_step]], axis=0)
            ##    print(f'Transaction  Shares held: {self.shares_held}')
            ##    print(f'Transaction  Virtual Shares held: {self.virtual_shares_held}')
            ##    print(f'Agente {self.i} metto in  vendita Shares: {int(self.virtual_shares_held * step_amount)}')
                self.virtual_shares_held -= int(self.virtual_shares_held * step_amount)
        self.net_worth = self.balance + self.shares_held * self.current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
      # print(f'Step Agent: {self.i}')
      # print(f'Shares held: {self.shares_held}')
      # print(f'Virtual Shares held: {self.virtual_shares_held}')
       # print(f'Action : {action}')
        self._take_action(action)
        if self.current_step > MAX_STEPS:
           # self.render()
            self.current_step = 0
        self.current_step += 1


        delay_modifier = (self.current_step / MAX_STEPS)
       # reward = np.exp(((self.balance - self.past_balance) / 1000)) * delay_modifier
        reward = (self.balance - self.past_balance) * delay_modifier

        #reward = self.balance * delay_modifier

        if self.i == 0 and self.balance > self.max_balance:
            self.max_balance = self.balance
            reward += 100
            print(f'Max Balance {self.max_balance}')
        done = self.balance >= INITIAL_ACCOUNT_BALANCE + 1000
        if done:
            print('Obiettivo raggiunto: ')
            self.render()
        obs = self._next_observation()
        self.past_balance = self.balance
        self.past_net_worth = self.net_worth
        self.balances.append(self.balance)
        self.shares_held_array.append(self.shares_held)

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        #print('REESETT')
        if self.i == 0:
            plt.figure(figsize=(10, 5))
            plt.title('Balance')
            plt.plot(range(0, len(self.balances)), self.balances)
            plt.show(block=False)
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.title('Shares held')
            plt.plot(range(0, len(self.shares_held_array)), self.shares_held_array)
            plt.show(block=False)
            plt.show()
        self.balances = []
        self.shares_held_array = []
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.virtual_balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 100
        self.virtual_shares_held = 100
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.ts = [[0, 0, 0, 0]]
        self.current_step = 0
        self.max_balance = INITIAL_ACCOUNT_BALANCE
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
     ##  print(f'Agent: {self.i}')
     ##  print(f'Current price: {self.current_price}')
     ##  print(f'Step: {self.current_step}')
     ##  print(f'Balance: {self.balance}')
     ##  print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
     ##  print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
     ##  print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
     ##  print(f'Profit: {profit}')
     ##  print(f'Transaction:')
     ##  print(self.ts)
    def step_wrapper(self, action, price, i, bids, asks, transaction, num):
        self.num = num
        self.bids = np.array(bids)
        self.i = i
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
                    self.ts = np.append(self.ts, [[transaction_action_type, item[4], transaction_shares, transaction_price]], axis=0)
                    self.transaction = np.delete(self.transaction, [j], 0)
                    j -= 1
                   # print(f'+RECOVERY shares held: {self.shares_held}')
                   # print(f'+RECOVERY virtual shares held: {self.virtual_shares_held}')
                   # print(f'+RECOVERY transaction shares : {transaction_shares}')
                    self.shares_held += transaction_shares
                    self.virtual_shares_held += transaction_shares
                    self.balance -= transaction_price * transaction_shares
                elif transaction_action_type >= 1 and transaction_action_type < 2:
                   # print(f'-RECOVERY shares held: {self.shares_held}')
                   # print(f'-RECOVERY virtual shares held: {self.virtual_shares_held}')
                   # print(f'-RECOVERY transaction shares : {transaction_shares}')
                   # print(f'-RECOVERY from agent : {transaction_agent_id}')
                   # print(f'-RECOVERY size transaction prima {self.transaction.size}')
                    self.shares_held -= transaction_shares
                    self.total_shares_sold += transaction_shares
                    self.balance += transaction_price * transaction_shares
                    self.virtual_balance += transaction_price * transaction_shares
                    self.ts = np.append(self.ts, [[transaction_action_type, item[4], transaction_shares, transaction_price]], axis=0)
                    self.transaction = np.delete(self.transaction, [j], 0)
                    j -= 1
                   # print(f'-RECOVERY size transaction dopo {self.transaction.size}')
            j += 1
        obs, rew, done, info = self.step(action)
        return obs, rew, done, info, self.bids, self.asks, self.current_price, self.transaction