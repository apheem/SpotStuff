import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_slimed.csv')
data.set_index('name', inplace=True)

data_feat = data.iloc[:,:-1].values
data_rank = data.iloc[:,-1].values


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

X = sc.fit_transform(data_feat)

from minisom import MiniSom
som = MiniSom(100,100, input_len=11)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers= ['o', 's']
colors= ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(X)
    plot(w[0]+0.5, w[1]+0.5, markers[data_rank[i]], 
         markeredgecolor=colors[data_rank[i]], 
         markerfacecolor= 'None',
         markersize=10,
         markeredgewidth=2)
show()
    