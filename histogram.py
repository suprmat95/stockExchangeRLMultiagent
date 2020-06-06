import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



colnames=['corr1', 'corr2', 'step']

data = pd.read_csv("pearson.csv", names=colnames)

print(data)


counts, bins = np.histogram(data['corr1'].values)
plt.figure()

plt.hist(bins[:-1], bins, weights=counts, label="Cor")
plt.axvline(data['corr1'].mean(), c='red', label='Mean')
print('Media:  \n')
print(data['corr1'].mean())
plt.axvline(data['corr1'].median(), c='red', linestyle='--', label='Median')
plt.legend(loc='upper right')
plt.show()

counts, bins = np.histogram(data['corr2'].values)
plt.figure()

plt.hist(bins[:-1], bins, weights=counts)
plt.axvline(data['corr2'].mean(), c='red', label = 'Mean')
plt.axvline(data['corr2'].median(), c='red', linestyle='--', label='Median')
plt.legend(loc='upper right')
plt.show()

counts, bins = np.histogram(data['step'].values)
plt.figure()

plt.hist(bins[:-1], bins, weights=counts)
plt.axvline(data['step'].mean(), c='red',label='Mean')
plt.axvline(data['step'].median(), c='red', linestyle='--', label='Median')
plt.legend(loc='upper right')
plt.show()