import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham


def CalGrid(user):
#user = pd.read_csv("user1.csv").sort_values(['utc'])
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

#    for i in range(len(track_x)):
#        plt.plot(track_x[i], track_y[i])
#    plt.show()
#    return track_x, track_y
    
    grid = np.zeros((int((x_max - x_min)/cell), int((y_max - y_min)/cell)))
    for i in range(len(track_x)):
        for j in range(len(track_x[i]) - 1):
	        last_x = int((track_x[i][j] - x_min)/cell)
	        last_y = int((track_y[i][j] - y_min)/cell)
	        next_x = int((track_x[i][j + 1] - x_min)/cell)
	        next_y = int((track_y[i][j + 1] - y_min)/cell)
	        result = list(bresenham(last_x, last_y, next_x, next_y))
#	        print last_x, last_y, next_x, next_y
	        if(len(result) == 1):
		        continue
	        else:
		        for k in result:
		            if(k[0] > 0 and k[0] < len(grid) and k[1] > 0 and k[1] < len(grid[0])):
			          grid[k[0]][k[1]] += 1
#	if(grid.sum() != 0):
#	    grid = grid/grid.sum()
    return track_x, track_y, grid

temp = pd.read_csv("top100.csv")

ids = np.load("ids.npy")

global x_min, x_max, y_min, y_max, cell, ext, x_range, x_range, x_label, y_label
x_min = 112000
x_max = 288000
y_min = 210000
y_max = 275000
cell = 1000
ext = [x_min/cell, x_max/cell, y_min/cell, y_max/cell]
x_range = range(120000/cell, 300000/cell, 20000/cell)
y_range = range(220000/cell, 280000/cell, 20000/cell)
x_label = ['120km', '140km', '160km', '180km', '200km', '220km', '240km', '260km', '280km']
y_label = ['220km', '240km', '260km', '280km']


track_x = []
track_y = []
#data = []
##data = np.zeros((int((x_max - x_min)/cell), int((y_max - y_min)/cell)))
#for i in ids:
#    print i
##grid = CalGrid(temp[temp['id']=='8f325f65d465d073a200a6c9d176b511867665fc5e182439c0f6d4116c7065a5'])
##    tr_x, tr_y, grid = CalGrid(temp[temp['id']=='8f325f65d465d073a200a6c9d176b511867665fc5e182439c0f6d4116c7065a5'].sort_values(['utc']))
#    tr_x, tr_y, grid= CalGrid(temp[temp['id']==i].sort_values(['utc']))
#    track_x = track_x + tr_x
#    track_y = track_y + tr_y
##    data = data + grid
#    data.append(np.reshape(grid, len(grid)*len(grid[0])))
    

tr_x, tr_y, grid= CalGrid(temp[temp['id']==ids[2]].sort_values(['utc']))
#tr_x, tr_y, grid= CalGrid(temp[temp['id']=='8f325f65d465d073a200a6c9d176b511867665fc5e182439c0f6d4116c7065a5'].sort_values(['utc']))
#track_x = tr_x
#track_y = tr_y
#for i in range(len(track_x)):
#    plt.plot(np.asarray(track_x[i])/cell, np.asarray(track_y[i])/cell, linewidth = 0.5)
#plt.xticks(x_range, x_label)
#plt.yticks(y_range, y_label)
#ax.set_ylim([y_min/cell, y_max/cell])
#ax.set_xlim([x_min/cell, x_max/cell])
#ax.set_title("Trajectory Map")
# for i in range(len(track_x)):
#plt.show()
tr_x, tr_y, grid= CalGrid(temp[temp['id']==ids[35]].sort_values(['utc']))
fig, ax = plt.subplots(figsize=(15,7))
#img = plt.imread("pr.JPG")
#ax.imshow(img, extent = ext, alpha=0.5)
im = ax.imshow(np.rot90(grid), cmap=plt.cm.Greys, alpha=0.7, extent = ext)
plt.xticks(x_range, x_label)
plt.yticks(y_range, y_label)
ax.set_ylim([y_min/cell, y_max/cell])
ax.set_xlim([x_min/cell, x_max/cell])
ax.set_title("Trajectory Map")
# for i in range(len(track_x)):
# 	plt.plot(np.asarray(track_x[i])/cell, np.asarray(track_y[i])/cell)
plt.show()
plt.scatter(np.asarray(tr_x[1])/cell, np.asarray(tr_y[1])/cell, s=1)
plt.plot(np.asarray(tr_x[1])/cell, np.asarray(tr_y[1])/cell, linewidth = 0.5)

plt.xticks(x_range, x_label)
plt.yticks(y_range, y_label)
ax.set_ylim([y_min/cell, y_max/cell])
ax.set_xlim([x_min/cell, x_max/cell])
ax.set_title("Trajectory Map")
plt.show()





