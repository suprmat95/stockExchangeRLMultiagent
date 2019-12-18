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
MAX_STEPS = 2000

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
            low=0, high=1, shape=(1 , 6), dtype=np.float16)

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
        #print('p_price: ')
        #print(action[2])
        #print('current price: ')
        #print(self.current_price)
        #print('price decurted: ')
        #print(self.current_price*p_price/100)
        #print('per')
        #print(p_price)
        #print('current price')
        #print(self.current_price)
        #print('movement price')
        movement_price = self.current_price + self.current_price*p_price/100
        #print(movement_price)
        #print('delta')
        #print(self.current_price * p_price / 100)
        find = True
        #print('movement price')
        #print(movement_price)
        #print('balance')
        #print(self.balance)
        total_possible = int(self.balance / movement_price)
        if action_type < 1:
            # Buy amount % of balance in shares
            #self.bids = np.append(self.bids, np.array([[self.i, action[0], action[1]]]), axis=0)
            #print('Compro')
            #print('asks number')
            #print(self.asks.size)
            #print('asks')
            #print(self.asks)
            # print(self.asks)
            if self.asks.size > 0:
                i = 0
                for item in self.asks:
                    if(item[0] != self.i ):
                        #print('item asks: ')
                        #print(float(item[0]))
                        #print(float(item[1]))
                        #print(float(item[2]))
                        #print(float(item[3]))

                        item_price = self.current_price + self.current_price*item[3]/100


                        #print('item_price: ')
                        #print(item_price)
                        #print('movement price')
                        #print(movement_price)
                        #print('item price')
                        #print(item_price)
                        if movement_price >= item_price:
                            ##print('trovato asks')

                            # print('itemprice ')
                           # print(item_price)
                            shares_bought = item[2]
                            prev_cost = self.cost_basis * self.shares_held
                            additional_cost = shares_bought * item_price
                            self.balance -= additional_cost
                            # print('compro: ', shares_bought, 'sheres held ', self.shares_held, ' balance: ', self.balance)
                            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                            self.shares_held += shares_bought
                           # print('current prima')
                           # print(self.current_price)
                            self.current_price = item_price
                            #print('curren dopo')
                            ##print(self.current_price)
                            #print('asks before: ')
                            #print(self.asks)
                            self.asks = np.delete(self.asks, [i], axis=0)
                            #print('asks after: ')
                            #print(self.asks)
                            np.append(self.transaction, np.array([[self.i, shares_bought * item_price]]))
                            find = False
                            break
                    i = i + 1
        if(find):
            # print('find')
             self.bids = np.append(self.bids, [[self.i, action[0], total_possible * amount, action[2]]], axis=0)
             #print('bids')
             #print(self.bids)

        elif action_type < 2:
            # Sell amount % of shares held
            # self.bids = np.append(self.bids, np.array([[self.i, action[0], action[1]]]), axis=0)
           # print('Vendo')
            find = True
           # print('bids number')
           # print(self.bids.size)
           # print('bids')
           # print(self.bids)
            if self.bids.size > 0:
                i = 0
                for item in self.bids:
                    if(item[0] != self.i ):

                        item_price = self.current_price + self.current_price*item[3]/100
                        if movement_price <= item_price:
                            #print('trovato bids: ')
                           # print('itemprice ')
                           # print(item_price)
                            #print('item bids: ')
                            #print(float(item[0]))
                            #print(float(item[1]))
                            #print(float(item[2]))
                            #print(float(item[3]))
                            shares_sold = item[2]
                            self.balance += shares_sold * item_price
                            #print('vendo: ', shares_sold, 'sheres held ', self.shares_held, ' balance: ', self.balance)
                            self.shares_held -= shares_sold
                            self.total_shares_sold += shares_sold
                            self.total_sales_value += shares_sold * item_price
                            #print('current prima')
                            #print(self.current_price)
                            self.current_price = item_price
                            ##print('curren dopo')
                            ##print(self.current_price)
                            #print('bids before: ')
                            #print(self.bids)
                            self.bids = np.delete(self.bids, [i], axis=0)
                            #print('bids after: ')
                            #print(self.asks)
                            #np.append(self.transaction, np.array([[self.i, shares_bought * item_price]]))
                            find = False
                            break
                    i = i + 1

        if (find):
            #print('find asks ')
            self.asks = np.append(self.asks, [[self.i, action[0], self.shares_held * amount, action[2]]], axis=0)
            self.shares_held -= int(self.shares_held * amount)
            self.total_shares_sold += int(self.shares_held * amount)
            #print('asks')
            #print(self.asks)


        self.net_worth = self.balance + self.shares_held * self.current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        print('action')
        print(action)
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
    def step_wrapper(self, action, price, i, bids, asks, transaction):
        #print('i: ', i)

        self.bids = np.array(bids)

        self.i = i

        self.asks = np.array(asks)

        self.current_price = price

        print('prima price')
        print(self.current_price)
        self.transaction = transaction

        obs, rew, done, info = self.step(action)

        print('dopo price')
        print(self.current_price)
        return obs, rew, done, info, self.bids, self.asks, self.current_price, self.transaction