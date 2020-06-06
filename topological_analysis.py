from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Scaler, Filtering, BettiCurve
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_point_cloud
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
from gtda.point_clouds import ConsistentRescaling
import numpy as np
import scipy.stats as st
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter
from gtda.time_series import Resampler, TakensEmbedding, SlidingWindow
import csv


colnames=['epoch', 'step', 'agent', 'balance', 'net_worth', 'shares', 'price', 'alpha']

data = pd.read_csv("prova1.csv", names=colnames)
print('data')
print(data)
data = data.loc[data['epoch'] == 2]
print('Data lenght')
print(data)
print(data.shape)
agent_as_index = data.set_index('agent')
agent_as_index.to_csv('out_agent.csv')
#print(agent_as_index)
steps_array = []
cloud_points = []
prices = []
alphas = []
for i in range(1, 1000):
    step = agent_as_index.loc[agent_as_index['step'] == i]
    prices.append(step.loc[0, 'price'])
    alphas.append(step.loc[0, 'alpha'])
   # print("Step")
   # print(step)
    for j in range(0, 10):
        agent = step.loc[j, :]
        #print('agent')
        #print(agent)
        steps_array.append([agent['balance'], agent['shares'], agent['net_worth'], agent['price']])
    #print('steps_array')
    #print(steps_array)
    cloud_points.append(steps_array)
    steps_array = []

VR = VietorisRipsPersistence()

cloud_points = np.asarray(cloud_points)[:, :, 2]
print('Cloud points')
print(cloud_points)
# Selecting the z-axis and the label rho
#X = cloud_points[:, 0:20:2]
X = cloud_points[:,1:10:2]
#X = cloud_points[:, :]
print('X')
print(X)
print('X size')
print(X.shape)
y = cloud_points[:, 1]
w = cloud_points[:, 2]
z = cloud_points[:, 3]

fig = px.line(title='Trajectory of the Lorenz solution, projected along the z-axis')
fig.add_scatter(y=X[:, 0], name='X')
fig.add_scatter(y=X[:, 1], name='u')
fig.add_scatter(y=X[:, 2], name='y')
fig.add_scatter(y=X[:, 3], name='r')
fig.add_scatter(y=X[:, 4], name='z')


#fig.show()

period = 10
periodicSampler = Resampler(period=period)

X_sampled, y_sampled = periodicSampler.fit_transform_resample(X, y)
Z_sampled, w_sampled = periodicSampler.fit_transform_resample(z, w)
print('X_sampled')
print(X_sampled)



fig = px.line(title='Trajectory of the Lorenz solution, projected along the z-axis and resampled every 10h')
fig.add_scatter(y=X_sampled.flatten(), name='X_sampled')
fig.add_scatter(y=y_sampled, name='y_sampled')
fig.add_scatter(y=Z_sampled.flatten(), name='Z_sampled')
fig.add_scatter(y=w_sampled, name='w_sampled')
#fig.show()

fig = px.line(title='Price')
fig.add_scatter(y=prices, name='price')
fig.show()

fig = px.line(title='Alpha')

fig.add_scatter(y=alphas, name='alphas')
fig.show()

#embedding_dimension = 10
#time_delay = 3
#TE = TakensEmbedding(parameters_type='search', dimension=embedding_dimension, time_delay=time_delay)

#TE.fit(X_sampled)
#time_delay_ = TE.time_delay_
#embedding_dimension_ = TE.dimension_
#X_embedded, y_embedded = TE.transform_resample(X_sampled, y_sampled)


#print('Optimal embedding time delay based on mutual information: ', time_delay_)
#print('Optimal embedding dimension based on false nearest neighbors: ', embedding_dimension_)

window_width = 40
window_stride = 10
SW = SlidingWindow(width=window_width, stride=window_stride)

X_windows = SW.fit_transform(X)
print('X_windows')
print(X_windows)
X_windows = SW.fit_transform(X)

print('X_windows')
print(X_windows)

window_number = 14
SW.plot(X_windows, sample=window_number)

embedded_begin, embedded_end = SW._slice_windows(X)[window_number]
window_indices = np.arange(embedded_begin, embedded_end)
fig = px.line(title=f'Resampled Lorenz solution over sliding window {window_number}')
fig.add_scatter(x=window_indices, y=X[window_indices], name='X_sampled')
fig.add_scatter(x=window_indices, y=X[window_indices], name='Y_sampled')
fig.show()

homology_dimensions = (0, 1, 2)
VR = VietorisRipsPersistence(
    metric='euclidean', max_edge_length=100, homology_dimensions=homology_dimensions)

X_diagrams = VR.fit_transform(X_windows)
plot_diagram(X_diagrams[window_number])

diagramScaler = Scaler()

X_scaled = diagramScaler.fit_transform(X_diagrams)

diagramScaler.plot(X_scaled, sample=window_number)

diagramFiltering = Filtering(epsilon=0.1, homology_dimensions=(1, 2))

X_filtered = diagramFiltering.fit_transform(X_scaled)

#diagramFiltering.plot(X_filtered, sample=window_number)

PE = PersistenceEntropy()

X_persistence_entropy = PE.fit_transform(X_scaled)

fig = px.line(title='Persistence entropies, indexed by sliding window number')
for dim in range(X_persistence_entropy.shape[1]):
    fig.add_scatter(y=X_persistence_entropy[:, dim], name=f'PE in homology dimension {dim}')
fig.show()
if(sum(X_persistence_entropy[:, 1]) !=0):
    X_persistence_entropy_d = [0] * window_width
    for i in X_persistence_entropy[:, 1]:
        X_persistence_entropy_d += [i]*10
    X_persistence_entropy_d = np.array(X_persistence_entropy_d)
    #X_betti_curves = BC.fit_transform(X_scaled)
    prices += [prices[len(prices)-1]]*1
    alphas += [alphas[len(alphas)-1]]*1
    prices_n = [p/max(prices) for p in prices]
    prices_d = np.diff(prices_n)
    alphas_n = [p/max(alphas) for p in alphas]

    alphas_d = np.diff(alphas)
    y_th = savgol_filter(prices_d, 51, 3)
    alphas_d_th = savgol_filter(alphas_d, 51, 3)
    X_persistence_entropy_n = [p/max(X_persistence_entropy_d) for p in X_persistence_entropy_d]
    X_persistence_entropy_n_d = np.diff(X_persistence_entropy_n)
    X_persistence_entropy_n_th = savgol_filter(X_persistence_entropy_n_d, 51, 3)
    print(f'alphas : {len(alphas)}')
    print(f'X_persistence_entropy_d : {len(X_persistence_entropy_n)}')
    corr_max = 0
    corr_max2 = 0
    i_max = 0

    for i in range(-100, 50):
        corr = st.pearsonr(np.roll(X_persistence_entropy_n_th, i), y_th)
        corr2 = st.pearsonr(np.roll(X_persistence_entropy_n_th, i), alphas_d_th)

        print(f'i: {i} corr: {corr}')
        print(f'i: {i} corr: {corr2}')

        if corr[0] > corr_max:
            i_max = i
            corr_max = corr[0]
        if corr2[0] > corr_max2:
            corr_max2 = corr2[0]
    print(f'price_d : {prices_d.shape}')
    print(f'X_persistence_entropy_d : {X_persistence_entropy_d.shape}')
    print(f'Corre: {corr_max}')
    print(f'i max: {i_max}')
    print(f'Corre 2: {corr_max2}')
    with open('correlazioni.csv', 'a+', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([corr_max, corr_max2])

    fig = px.line(title='Persistence entropies, indexed by sliding window number')
    #fig.add_scatter(y=X_persistence_entropy_n, name='PE in homolgy dimension 0')
    fig.add_scatter(y=X_persistence_entropy_n_th, name='derived pe1')

    #fig.add_scatter(y=prices_n, name='prices')
    #fig.add_scatter(y=alphas, name='alpha')
    #fig.add_scatter(y=prices_d, name='derivata')
    fig.add_scatter(y=y_th, name='derived price')
    fig.add_scatter(y=alphas_d_th, name='derived alpha')

    fig.show()

    #BC.plot(X_betti_curves, sample=window_number)


    #print('Cloud points: ', np.asarray(cloud_points)[:,:,1])

    #plot_point_cloud(np.asarray(cloud_points)[:,:,1])
    #cloud_points = np.array([cloud_points])
    #diagrams = VR.fit_transform(np.asarray(cloud_points))
    #print('Diagrams')
    #print(diagrams.shape)
    #print('shape cloud point: ')
    #print(plot_diagram(diagrams[780,:, :]))
    #with open('test.txt', 'a+') as outfile:
    #    for slice_2d in diagrams:
            #print(slice_2d)
            #plot_point_cloud(np.asarray(slice_2d))
    #        np.savetxt(outfile, slice_2d)
    #        outfile.write('________________\n')
    #PE = PersistenceEntropy()
    #features = PE.fit_transform(diagrams)
    #print('Features shapes: ')
    #print(features.shape)
    #print(features)
    #for i in range(1, 497):
        #print(features[0])
    #plt.figure(figsize=(15, 5))
    #plt.xlabel("Steps ")
    #plt.ylabel("Value")
    #df_balances = pd.DataFrame(dict(H0=features[:, 0], H1=features[:, 1], price=prices))
    #plt.title('Balances')
    #plt.plot(range(0, df_balances.to_numpy().shape[0]), df_balances.to_numpy(), label='H0')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.show(block=False)
    #plt.show()