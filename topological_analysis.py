from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import PersistenceEntropy
import numpy as np

import pandas as pd
colnames=['epoch', 'step', 'agent', 'balance', 'net_worth', 'shares', 'price']

data = pd.read_csv("prova1.csv", names=colnames)
data = data.loc[data['epoch'] == 5]
agent_as_index = data.set_index('agent')
agent_as_index.to_csv('out_agent.csv')
print(agent_as_index)
steps_array = []
cloud_points = []
for i in range(1, 497):
    step = agent_as_index.loc[agent_as_index['step'] == i]
  #  print("Step")
  #  print(step)
    for j in range(0, 10):
        agent = step.loc[j, :]
    #    print('agent')
    #    print(agent)
        steps_array.append([agent['balance'], agent['net_worth'], agent['shares']])
  #  print('steps_array')
  #  print(steps_array)
    cloud_points.append(steps_array)
    steps_array = []
VR = VietorisRipsPersistence()

print('Cloud points: ', np.asarray(cloud_points))
#cloud_points = np.array([cloud_points])
diagrams = VR.fit_transform(np.array(cloud_points))
print('diagrams: ')

with open('test.txt', 'a+') as outfile:
    for slice_2d in diagrams:
        print(slice_2d)
        np.savetxt(outfile, slice_2d)
        outfile.write('________________\n')
PE = PersistenceEntropy()
features = PE.fit_transform(diagrams)
print('Features: ')
print(features)
print(len(features))
