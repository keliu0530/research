import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


x_min = 245000
x_max = 260000
y_min = 240000
y_max = 265000

user = pd.read_csv("user1.csv").sort_values(["utc"])

utc = user['utc'].as_matrix()

user['hour'] = ((utc - 1503619200)%86400/3600)

clusters = []
#user['hour'] = range(len(user))
#fig, ax = plt.subplots(figsize=(7,7))
#for i in [20]:
#for i in user.hour.unique():
##    x = user[user.hour == i]['x'].as_matrix()
##    y = user[user.hour == i]['y'].as_matrix()
#    X = user[user.hour == i][['x','y']].as_matrix()
#    db = DBSCAN(eps=1000, min_samples=10).fit(X)
#    labels = db.labels_
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    clusters.append(n_clusters_)
    
interval = 600
for i in range(0, 86400, interval):
    X = user[(user.hour >= i) & (user.hour < i + interval)][['x','y']].as_matrix()
    db = DBSCAN(eps=1000, min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters.append(n_clusters_)
    
#    plt.plot(x,y, '.')
    
#ax.set_ylim([y_min, y_max])
#ax.set_xlim([x_min, x_max])
#plt.show()

#X = user[user.hour == 13][['x', 'y']].as_matrix()

##X = user[['x', 'y']].as_matrix()

#db = DBSCAN(eps=1000, min_samples=10, algorithm='kd_tree', leaf_size=30, p=None, n_jobs=-1).fit(X)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
#labels = db.labels_

## Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

## Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels))]
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]

#    class_member_mask = (labels == k)

#    xy = X[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)

#    xy = X[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)

#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
