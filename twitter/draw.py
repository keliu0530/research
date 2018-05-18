#!/usr/bin/python
import MySQLdb
from os import listdir
import pandas as pd
import utm
import numpy as np
from pyproj import Proj, transform
import os.path
import scipy.stats as stats
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
from matplotlib.patches import Circle
from math import sqrt 
from scipy import signal
import statsmodels.api as sm

x_min = 910600
x_max = 915700
y_min = 571700
y_max = 582000

global x_grid, y_grid, W, cell, x_range, x_range, x_label, y_label, mask_U, mask_V, dist, values, font
font = 20
cell = 10
x_grid = range(x_min + cell/2, x_max + cell/2, cell)
y_grid = range(y_min + cell/2, y_max + cell/2, cell)
ext = [x_min/cell, x_max/cell, y_min/cell, y_max/cell]
W = 2
x_range = range(911000/cell, 917000/cell, 2000/cell)
y_range = range(572000/cell, 584000/cell, 2000/cell)
x_label = ['911km', '913km', '915km']
y_label = ['572km', '574km', '576km', '578km', '580km', '582km']

temp = []
for i in range(-W, W+1):
    temp.append(-1 * i * np.ones(W * 2 + 1))
mask_V = np.array(temp)
del temp
mask_U = mask_V.T
#mask_U = - mask_U
dist = mask_U**2 + mask_V**2
dist[W][W] = 1

data = pd.read_csv("Mar13.csv")


t1_1 = 1394721000
t1_2 = t1_1 + 3600

t2_1 = 1394728200
t2_2 = t2_1 + 3600
#prob1 = kdemap(t1_1, t1_2)

t_start = t2_1
t_end = t2_2

#t_start = t1_1
#t_end = t1_2

values = data[(data['utc'] >= t_start) & (data['utc'] < t_end)].as_matrix(["x", "y"]).T
print "Finish retrieve data"
kde = sm.nonparametric.KDEMultivariate(values.T, var_type='cc',bw='normal_reference')  
prob = kde.pdf([np.meshgrid(x_grid, y_grid)[0].T.ravel(), np.meshgrid(x_grid, y_grid)[1].T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell * cell

#kde = stats.gaussian_kde(values, bw_method='scott')
#prob = kde.evaluate([np.meshgrid(x_grid, y_grid)[0].T.ravel(), np.meshgrid(x_grid, y_grid)[1].T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell * cell
print "Finish getting probability"


period = time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_start)) + '-' + time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_end)).split(' ')[1]
#title = 'Kernel Density Estimation\n' + period
title = 'Heat Map\n' + period
name = 'pics/kde' + str(t_start) + '-' + str(t_end) + '-' + str(cell) + '.png'
#DrawKDE(title, name, prob)


fig, ax = plt.subplots(figsize=(10,15))
img = plt.imread("wl.JPG")
ax.imshow(img, extent = ext)
im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
plt.plot(values[0]/(cell*1.0), values[1]/(cell*1.0),'ko',ms=1)
plt.xticks(x_range, x_label)
plt.yticks(y_range, y_label)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.3)
ax.set_title(title)
cbar = plt.colorbar(im, cax=cax, format='%.0e')
#cbar.set_label('Probability')

for font_objects in cbar.ax.yaxis.get_ticklabels():
    font_objects.set_size(font)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font)
#fig.savefig(name)   # save the figure to file
print name + " saved..."
plt.show()
