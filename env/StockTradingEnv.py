import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 5242

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StockTradingEnv, self).__init__()
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1]), high=np.array([3, 1, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 6), dtype=np.float16)

        self.past_balance = 0




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
        action_type = action[0]
        amount = action[1]
        p_price = action[2]
        movement_price = self.current_price + self.current_price*p_price/100
        find = True
        total_possible = int(self.balance / movement_price)
        if action_type <= 1:
            # Buy amount % of balance in shares
           # print('Compro')
            if self.asks.size > 0:
                j = 0
                for item in self.asks:
                    if item[0] != self.i:
                        item_price = item[3]
                        if movement_price >= item_price and self.balance >= item[3]*item[2] and find:
                           # print('TROVATO ASK: ')
                           # print(action_type)
                            shares_bought = item[2]
                            prev_cost = self.cost_basis * self.shares_held
                            additional_cost = item_price
                            self.balance -= additional_cost
                            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                            self.shares_held += shares_bought
                            self.current_price = item_price
                            self.asks = np.delete(self.asks, j, axis=0)
                            self.transaction = np.append(self.transaction, [[self.i, action_type, shares_bought,  item_price]], axis=0)
                            find = False
                            break
                    j = j + 1
        elif action_type> 1 and action_type <= 2:
           ## print('VENDO')
            if self.bids.size > 0:
                j = 0
                for item in self.bids:
                    if(item[0] != self.i ):
                        item_price = item[3]
                        shares_sold = item[2]
                        if movement_price <= item_price and self.shares_held >= shares_sold and find:
                            #print('TROVATO BIDS: ')
                            #print('bid:')
                            #print(item)
                            self.balance += item_price
                            self.shares_held -= shares_sold
                            self.total_shares_sold += shares_sold
                            self.total_sales_value += item_price
                            self.current_price = item_price
                            self.bids = np.delete(self.bids, j, 0)
                            self.transaction = np.append(self.transaction, [[self.i, action_type, shares_sold,  item_price]], axis=0)
                            find = False
                            break
                    j = j + 1

        if (find):
            if action_type < 1:
                self.bids = np.append(self.bids, [[self.i, action_type, int(total_possible * amount), movement_price]], axis=0)
                self.balance -= movement_price * int(total_possible * amount)
            else:
                self.asks = np.append(self.asks,[[self.i, action_type, int(self.shares_held * amount), movement_price]], axis=0)
                self.shares_held -= int(self.shares_held * amount)
        self.net_worth = self.balance + self.shares_held * self.current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        #print('action')
        #print(action)
        self._take_action(action)
        self.current_step += 1
        if self.current_step > 5242:
            self.current_step = 0
        delay_modifier = (self.current_step / MAX_STEPS)
        reward = (self.balance - self.past_balance) * delay_modifier
        done = self.balance > 20000
        obs = self._next_observation()
        self.past_balance = self.balance
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 100
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, 5242)

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
    def step_wrapper(self, action, price, i, bids, asks, transaction, current_step):
        #print('i: ', i)

        self.current_step = current_step

        self.bids = np.array(bids)

        self.i = i

        self.asks = np.array(asks)

        self.current_price = price



        self.transaction = transaction
        j = 0
        #print('transaction size: ')
        #print(self.transaction.size)
        #print('transaction')
        #print(self.transaction)
        j = 0
        for item in self.transaction:
            agent_id = item[0]
            action_type = item[1]
            movement_price = item[3]
            shares = item[2]
            if agent_id == self.i:
                #print('entrato')
                if action_type <= 1:
                    self.transaction = np.delete(self.transaction,[j], 0)
                    self.shares_held += shares

                elif action_type > 1 and action_type <= 2:
                    self.total_shares_sold += shares
                    self.balance += movement_price*shares
                    self.transaction = np.delete(self.transaction, [j], 0)

            j += 1

        #print('_______________________________')
        #print('prima price')
        #print(self.current_price)
        obs, rew, done, info = self.step(action)
        #print('dopo price')
        #print(self.current_price)
        #print('balance')
        #print(self.balance)
        #print('_______________________________')


        return obs, rew, done, info, self.bids, self.asks, self.current_price, self.transaction