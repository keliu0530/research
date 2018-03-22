import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham

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
interval2 = []
for i in range(len(gap) - 1):
	if(gap[i + 1] - 1 - gap[i] > 0):
		track_x.append(x[gap[i] : gap[i + 1] - 1])
		track_y.append(y[gap[i] : gap[i + 1] - 1])
		interval2 = interval2 + interval[gap[i] + 1 : gap[i + 1] - 1].tolist()

# for i in range(len(track_x)):
# 	plt.plot(track_x[i], track_y[i])
# plt.show()

x_min = 112000
x_max = 288000
y_min = 210000
y_max = 275000

cell = 100
grid = np.zeros((int((x_max - x_min)/cell), int((y_max - y_min)/cell)))
ext = [x_min/cell, x_max/cell, y_min/cell, y_max/cell]

for i in range(len(track_x)):
	for j in range(len(track_x[i]) - 1):
		last_x = int((track_x[i][j] - x_min)/cell)
		last_y = int((track_y[i][j] - y_min)/cell)
		next_x = int((track_x[i][j + 1] - x_min)/cell)
		next_y = int((track_y[i][j + 1] - y_min)/cell)
		result = list(bresenham(last_x, last_y, next_x, next_y))
		if(len(result) == 1):
			continue
		else:
			for k in result:
				grid[k[0]][k[1]] += 1

fig, ax = plt.subplots(figsize=(15,7))
img = plt.imread("pr.JPG")
ax.imshow(img, extent = ext, alpha=0.5)
im = ax.imshow(np.rot90(grid), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
# for i in range(len(track_x)):
# 	plt.plot(np.asarray(track_x[i])/cell, np.asarray(track_y[i])/cell)
plt.show()
