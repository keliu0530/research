import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

user = pd.read_csv("user1.csv").sort_values(['utc'])

record = user.utc.values

record = np.insert(record, 0, user.iloc[0].utc)

record = np.delete(record, len(record) - 1)

interval = (user.utc.values - record)

del record

x = user['x'].values.tolist()

y = user['y'].values.tolist()

gap = np.where(interval > 3600)[0]
gap = np.insert(gap, 0, 0)
if gap[-1] != len(user) - 1:
	gap = np.insert(gap, -1, len(user) - 1)

track_x = []
track_y = []
for i in range(len(gap) - 1):
	if(gap[i + 1] - 1 - gap[i] > 0):
		track_x.append(x[gap[i] : gap[i + 1] - 1])
		track_y.append(y[gap[i] : gap[i + 1] - 1])

for i in range(len(track_x)):
	plt.plot(track_x[i], track_y[i])
plt.show()