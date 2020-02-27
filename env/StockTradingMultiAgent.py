import random
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
import numpy as np
from env.StockTradingEnv import StockTradingEnv
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting

MAX_NUM_SHARES = 2147483647


class StockTradingMultiAgent(MultiAgentEnv):

    def __init__(self, num):
        super(StockTradingMultiAgent, self).__init__()
        self.agents = [StockTradingEnv() for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.price = random.randint(1, 5)
        self.step_current = self.price
        self.num = num
        self.df_net_worthes = pd.DataFrame()
        self.alpha = np.random.normal(0, 0.05, 1)[0]
        self.asks = np.empty((0, 5))
        self.bids = np.empty((0, 6))
        self.steppps = 0
        self.alphas = []
        self.prices = []
        self.transaction = np.empty((0, 4))

    def reset(self):
      #  print('RESSSETTTTT')
        self.steppps = 0
        self.asks = np.empty((0, 5))
        self.bids = np.empty((0, 6))
        self.price = random.randint(1, 5)
        self.dones = set()
        self.alpha = np.random.normal(0, 0.05, 1)[0]
        self.prices = []
        self.transaction = np.empty((0, 4))
        self.alphas = []
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info, quantity, net_worthes, balances, shares_held = {}, {}, {}, {}, {}, {}, {}, {}
        done["__all__"] = False
        if self.price <= 0.1:
            print('Gioco finito')
            done["__all__"] = True
            self.reset()
        #print(f' self price: {self.price}')
        random_steps = [30, 40, 50]
        if self.price > 0.1:
            if self.steppps % random_steps[random.randint(0, 2)] == 0:
                self.price = self.price + (self.price * self.alpha)/2
                self.alpha += np.random.normal(0, 0.1, 1)[0]
                self.alphas.append(self.alpha)
            for i, action in action_dict.items():
                if np.isnan(action).any() == False:
                    print(f'i: {i}')
                    self.bids, self.asks = self.agents[i].bet_an_offer_wrapper(action, i, self.bids, self.asks, self.price)
            self._solve_book()
            i = 0
            tot = 0
            tot_qty = 0
            price_next_step = 0
            for item in self.transaction:
                tot+= item[2]*item[3]
                tot_qty += item[2]
                price_next_step = tot / tot_qty
            for i in range(0, len(self.agents)):
                obs[i], rew[i], done[i], info[i], net_worthes[i], balances[i], shares_held[i], self.transaction = self.agents[i].step_wrapper(self.price, self.transaction, i)
            if price_next_step != 0:
                self.price = price_next_step
            self.steppps += 1

            del self.asks
            del self.bids
            del self.transaction
            self.asks = np.empty((0, 5))
            self.bids = np.empty((0, 6))
            self.transaction = np.empty((0, 4))

        if self.steppps > 700:
            self.df_net_worthes = pd.DataFrame(net_worthes)
            #Show Price
            plt.figure(figsize=(10, 5))
            plt.title('Price')
            plt.xlabel("Steps ")
            plt.ylabel("Value")
            plt.plot(range(0, len(self.prices)), self.prices)
            plt.show(block=False)
            plt.show()
            # Show Alpha
            plt.figure(figsize=(10, 5))
            plt.title('alpha')
            plt.xlabel("Steps ")
            plt.ylabel("Value")
            plt.plot(range(0, len(self.alphas)), self.alphas)
            plt.show(block=False)
            plt.show()
            #Show Net worths
            plt.figure(figsize=(15, 5))
            plt.title('Net worthes')
            plt.plot(range(0, self.df_net_worthes.to_numpy().shape[0]), self.df_net_worthes.to_numpy(), label ='Agent')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show(block=False)
            plt.show()
            # Show Balances
            plt.figure(figsize=(15, 5))
            plt.xlabel("Steps ")
            plt.ylabel("Value")
            df_balances = pd.DataFrame(balances)
            plt.title('Balances')
            plt.plot(range(0, df_balances.to_numpy().shape[0]), df_balances.to_numpy(), label='Agent')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show(block=False)
            plt.show()
            # Show Shares Held
            plt.figure(figsize=(15, 5))
            df_sharesheld = pd.DataFrame(shares_held)
            plt.title('Shares held')
            plt.xlabel("Steps ")
            plt.ylabel("Value")
            plt.plot(range(0, df_sharesheld.to_numpy().shape[0]), df_sharesheld.to_numpy(), label='Agent')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show(block=False)
            plt.show()
            #Net Worth comparision
            plt.figure(figsize=(10, 5))
            x = 3
            y = 4
            plt.title('Net_Worth Comparision')
            plt.xlabel(f"Agent {x}")
            plt.ylabel(f"Agent {y}")
            colors = range(0, self.df_net_worthes[0].size)
            plt.scatter(self.df_net_worthes[x], self.df_net_worthes[y], c=colors, cmap='Greens')
            cbar = plt.colorbar()
            plt.show(block=False)
            plt.show()
            #PCA
            pca = PCA(n_components=3)
            pca.fit(self.df_net_worthes.to_numpy())
            x_pca = pca.transform(self.df_net_worthes.to_numpy())
            #Show PCA 2
            plt.figure(figsize=(10, 5))
            plt.title("PCA 2")
            plt.xlabel(f"PCA 1")
            plt.ylabel(f"PCA 2")
            plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors, cmap='Greens')
            cbar.set_label('Steps')
            plt.show(block=False)
            plt.show()
            # Show PCA 3
            fig = plt.figure(figsize=(10, 5))
            plt.title('PCA 3')
            ax = fig.gca(projection='3d')
            colors = range(0, self.df_net_worthes[0].size)
            ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=colors, cmap='Greens')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
            done["__all__"] = True

        return obs, rew, done, info

    #def book_solver(self,bids, asks):
    def _solve_book(self):
       i = 0
       j = 0
       while i < len(self.bids) and j < len(self.asks):
          # print('Entrato')
           transaction_price = (self.bids[i][3] + self.asks[j][3])/2
           buyer = self.bids[i][0]
           seller = self.asks[j][0]
           if self.bids[i][2] > self.asks[j][2]:
               quantity = self.asks[j][2]
               self.bids[i][2] -= self.asks[j][2]
               self.transaction = np.append(self.transaction, [[buyer, 0, quantity, transaction_price],[seller, 1, quantity, transaction_price]], axis=0)
               self.asks = np.delete(self.asks, [j], axis=0)
               j += 1
           elif self.bids[i][2] < self.asks[j][2]:
               quantity = self.bids[i][2]
               self.asks[j][2] -= self.bids[i][2]
               self.transaction = np.append(self.transaction, [[buyer, 0, quantity, transaction_price],[seller, 1, quantity, transaction_price]], axis=0)
               self.bids = np.delete(self.bids, [i], axis=0)
               i += 1
           elif self.bids[i][2] == self.asks[j][2]:
               quantity = self.bids[i][2]
               self.transaction = np.append(self.transaction, [[buyer, 0,quantity, transaction_price], [seller, 1, quantity, transaction_price]], axis=0)
               self.asks = np.delete(self.asks, [j], axis=0)
               self.bids = np.delete(self.bids, [i], axis=0)
               i += 1
               j += 1

    def _reset_array(self, array):
        N = array.shape[0]
        for i in range(0,N):
            array = np.delete(array, [0], axis=0)
        return array













