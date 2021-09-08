import random as random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
LENGTH = 900
def shifted_correlations(pointsx, pointsy):
  correl = []
  for shift in range(-40,40):
    if shift <= 0:
      cov = np.array([pointsx[-shift:LENGTH], pointsy[0:LENGTH+shift]])
      correl.append(np.corrcoef(cov)[0,1])
    else:
      cov = np.array([pointsx[0:LENGTH-shift], pointsy[shift:LENGTH]])
      correl.append(np.corrcoef(cov)[0,1])
  return correl
print('Sample:\n')
print(np.random.choice([0,2], 3, replace=True))
max_correl = []
data1 = pd.read_csv("time_series_entropia.csv")
data2 = pd.read_csv("time_series_momentum.csv")
for x in range(10000):
  pointsx= random.sample(list(np.array(data1).flatten()), LENGTH)
  pointsy= random.sample(list(np.array(data2).flatten()), LENGTH)
  max_correl.append(max(shifted_correlations(pointsx, pointsy)))
mean = []
n = 0
for i in range(1, 10000):
  mean.append(np.mean(np.random.choice(max_correl, 10000, replace=True)))
print('mean:\n')
print(mean)
for i in mean:
  if i>0.16:
    n+=1
print('n\n')
print(n)
print('percent:\n')
print(n/10000)
plt.hist(max_correl, bins = 20)
plt.show()