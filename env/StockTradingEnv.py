import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_ACCOUNT_BALANCE = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 1000
INITIAL_ACCOUNT_BALANCE = 1000
INITIAL_NUMBER_SHARES = 100

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n):
        super(StockTradingEnv, self).__init__()
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, -1]), high=np.array([3, 0.5, 1]), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10, 5), dtype=np.float16)
        self.net_worthes = []
        self.rewards = []
        self.balances = []
        self.shares_held_array = []
        self.virtual_balances = []
        self.virtual_shares_held_array = []
        self.current_action = 0
        self.asks_columns_name = ['Agent', 'Action_Type', 'Shares', 'Step_Price', 'Current_Step']
        self.bids_columns_name = ['Agent', 'Action_Type', 'Shares', 'Step_Price', 'Current_Step', 'Share_Price']

        self.max_balance = INITIAL_ACCOUNT_BALANCE
        self.MAX_NUM_SHARES = n * INITIAL_NUMBER_SHARES

        self.num = 0
        self.i = 0
        self.initial_net_worth = 0
        self.past_balance = 0
        self.past_net_worth = 0
        self.current_price = 0
        self.frame = np.zeros((10, 5))

    def _next_observation(self):

        self.frame = np.delete(self.frame, [0], axis=0)
        self.frame = np.append(self.frame,
                               [[self.current_price / 700,
                                 self.balance / MAX_ACCOUNT_BALANCE,
                                 self.max_net_worth / MAX_ACCOUNT_BALANCE,
                                 self.shares_held / self.MAX_NUM_SHARES,
                             #    self.cost_basis / MAX_SHARE_PRICE,
                             #    self.total_shares_sold / self.MAX_NUM_SHARES,
                                 self.total_sales_value / (self.MAX_NUM_SHARES * MAX_SHARE_PRICE),
                                 ]], axis=0)
        return np.array(self.frame)

    def _bet_an_offer(self, action):
        # Set the current price to a random price within the time step
        step_action_type = action[0]
        step_amount = action[1]
        step_percent_price = action[2]
        step_price = self.current_price + self.current_price*step_percent_price/100
       # print(f'current price: ', self.current_price,' step price: ', step_price)
        step_total_possible = int(self.virtual_balance / step_price)
        step_bought_shares = int(step_amount * step_total_possible)
        step_price_shares = step_price * step_bought_shares
        step_shares_bought_min = step_bought_shares - (step_bought_shares * 0.2)
        step_shares_bought_max = step_bought_shares + (step_bought_shares * 0.2)
        step_sold_shares = int(step_amount * self.shares_held)
        step_shares_sold_min = step_sold_shares - (step_sold_shares * 0.2)
        step_shares_sold_max = step_sold_shares + (step_sold_shares * 0.2)
        bids = pd.DataFrame(columns=self.bids_columns_name)
        asks = pd.DataFrame(columns=self.asks_columns_name)
        if step_bought_shares > 0 and step_action_type < 1 and self.virtual_balance >= step_price_shares:
            bids = bids.append(pd.DataFrame(np.array([[self.i, step_action_type, step_bought_shares, step_price, self.current_step, step_price_shares]]), columns=self.bids_columns_name), ignore_index=True)
               # print('Find section')
               # print(f'Virtual balance: {self.virtual_balance} subtraction: {step_price * int(step_total_possible * step_amount)}')
            self.virtual_balance -= step_price_shares

        elif 0 < step_sold_shares <= self.virtual_shares_held and 1 <= step_action_type < 2:
            asks = asks.append(pd.DataFrame(np.array([[self.i, step_action_type, step_sold_shares, step_price, self.current_step]]), columns=self.asks_columns_name), ignore_index=True)
            self.virtual_shares_held -= step_sold_shares
        self.net_worth = self.balance + self.shares_held * self.current_price
        if self.current_step == 0:
            self.initial_net_worth = self.net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        if self.shares_held == 0:
            self.cost_basis = 0

        return bids, asks

    def step(self, action):
        delay_modifier = (self.current_step / MAX_STEPS)
        reward = (2 * self.net_worth - self.initial_net_worth - self.past_net_worth) * delay_modifier
       # print(f'Reward: ',reward )
        if self.current_step > MAX_STEPS:
            # self.render()
           # if self.i == 0 or self.i == 1:
           #     self.render()
            self.current_step = 0
        self.current_step += 1
       # reward = - (self.net_worth - self.past_net_worth)
        done = self.balance >= INITIAL_ACCOUNT_BALANCE*5
        #if done:
            #print(f'belence: {self.balance} Initial: {INITIAL_ACCOUNT_BALANCE} ')
            #self.render()
        obs = self._next_observation()
        self.past_balance = self.balance
        self.past_net_worth = self.net_worth
        self.net_worthes.append(self.net_worth)
        self.rewards.append(reward)
        self.balances.append(self.balance)
        self.shares_held_array.append(self.shares_held)
        self.virtual_balances.append(self.virtual_balance)
        self.virtual_shares_held_array.append(self.virtual_shares_held)
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        if self.i == 4:
             plt.figure(figsize=(10, 5))
             plt.title(f'Reward {self.i}')
             plt.xlabel("Steps ")
             plt.ylabel("Value")
             plt.plot(range(0, len(self.rewards)), self.rewards)
             plt.show(block=False)
             plt.show()
        self.net_worthes = []
        self.rewards = []
        self.balances = []
        self.shares_held_array = []
        self.virtual_balances = []
        self.virtual_shares_held_array = []
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.virtual_balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_balance = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 100
        self.virtual_shares_held = INITIAL_NUMBER_SHARES
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_action = 0
        self.current_step = 0
        self.frame = np.zeros((10, 5))

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Agent: {self.i}')
        print(f'Current price: {self.current_price}')
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        print(f'Transaction:')
        #print(self.ts)

    def step_wrapper(self, price, transaction, i):
        self.i = i
        #print(f'i: {i}')
        #print('tr')
        #print(transaction)
        self.current_price = price
        if not transaction.empty:
            transaction = self.complete_transaction(transaction)
        obs, rew, done, info = self.step(self.current_action)
        return obs, rew, done, info,  self.net_worthes, self.balances, self.shares_held_array,  transaction

    def bet_an_offer_wrapper(self, action, i, bids, asks, price):
        self.virtual_balance = self.balance
        self.virtual_shares_held = self.shares_held
        self.current_action = action
        self.bids = np.array(bids)
        self.asks = np.array(asks)
        self.i = i
        self.current_price = price
        return self._bet_an_offer(action)

    def complete_transaction(self, transaction):
        #print('_______________________________')
        #print('transaction')
        #print(transaction)
        transaction_item = transaction[(transaction['Agent'] == self.i)]
        if not transaction_item.empty:
            sold = transaction_item[(transaction_item['Action_Type'] < 1)]
            #print('Compro')
            #print(sold)
            if not sold.empty:
                sum = sold['Share'].sum()
                #print(f'Tot shares comprate: {sum}')
                self.shares_held += sum
                self.virtual_shares_held += sum
                self.balance -= (sold['Share'] * sold['Price']).sum()
                self.virtual_balances.append(self.virtual_balance)
                self.virtual_shares_held_array.append(self.virtual_shares_held)
                transaction.drop(sold.index, inplace=True)
            bought = transaction_item[(transaction_item['Action_Type'] >= 1) & (transaction_item['Action_Type'] < 2)]
            #print('Vendo')
            #print(bought)
            if not bought.empty:
                sum = bought['Share'].sum()
                #print(f'Tot shares vendute: {sum}')
                self.shares_held -= sum
                self.total_shares_sold += sum
                self.total_sales_value += (bought['Share'] * bought['Price']).sum()
                self.balance += (bought['Share'] * bought['Price']).sum()
                self.virtual_balances.append(self.virtual_balance)
                self.virtual_shares_held_array.append(self.virtual_shares_held)
                transaction.drop(bought.index, inplace=True)
        return transaction