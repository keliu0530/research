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

x_min = 112000
x_max = 288000
y_min = 210000
y_max = 275000

global x_grid, y_grid, W, t1, t2, cell, x_label, y_label
cell = 1000
x_grid = range(x_min + cell/2, x_max + cell/2, cell)
y_grid = range(y_min + cell/2, y_max + cell/2, cell)
ext = [x_min/cell, x_max/cell, y_min/cell, y_max/cell]
W = 2
x_label = ['120km', '140km', '160km', '180km', '200km', '220km', '240km', '260km', '280km']
y_label = ['220km', '240km', '260km', '280km']



def kdemap(t_start, t_end):
    if("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy" in os.listdir("./")):
        prob = np.load("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy")
        return prob
    if(str(t_start) + "-" + str(t_end) + ".npy" in os.listdir("./")):
        values = np.load(str(t_start) + "-" + str(t_end) + ".npy")
    else:
        db = MySQLdb.connect(host="keliudb.corzaau5yusv.us-east-2.rds.amazonaws.com",    # your host, usually localhost
                             user="keliu",         # your username
                             passwd="19930530",  # your password
                             db="PuertoRico")        # name of the data base

        df = pd.read_sql("SELECT x, y FROM PhoneData WHERE utc >= " + str(t_start) + " and utc < " + str(t_end), con=db)
        db.close()
        values = df.as_matrix(["x", "y"]).T
        del df
        np.save(str(t_start) + "-" + str(t_end), values)
#        return values
    print "Finish reading data..."
    
    period = time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_start)) + '-' + time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_end)).split(' ')[1]

    #def kde(t1):
    kde = stats.gaussian_kde(values)
    prob = kde.evaluate([np.meshgrid(x_grid, y_grid)[0].T.ravel(), np.meshgrid(x_grid, y_grid)[1].T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell *cell

    print "Finish calculating kde..."
    #
    fig, ax = plt.subplots(figsize=(15,10))
    img = plt.imread("pr.JPG")
    ax.imshow(img, extent = ext)
    im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
    plt.xlabel('x(km)')
    plt.ylabel('y(km)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    ax.set_title('Kernel Density Estimation(' + period + ')')
    plt.colorbar(im, cax=cax)

    #cbar = fig.colorbar(cax)
    fig.savefig('figures/kde' + str(t_start) + '-' + str(t_end) + '-' + str(cell) + '.png')   # save the figure to file
    print "KDE map saved..."
    np.save("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy", prob)
    return prob


#t1 = kdemap(1503916200, 1503919800) #8.28.2017 6:30am to 7:30am
t1 = kdemap(1503948600, 1503952200) #8.28.2017 3:30am to 4:30pm
print "Done with t1..."
#t2 = kdemap(1503927000, 1503930600) #8.28.2017 9:30am to 10:30am
t2 = kdemap(1503959400, 1503963000) #8.28.2017 6:30pm to 7:30pm
print "Done with t2..."

def FMap(x, y):
    t = t2 - t1
    vector = np.zeros(2)
    for p in range(-W, W+1):
        for q in range(-W, W+1):
            if(sqrt(p*p + q*q) != 0):
#                vector = vector + np.array([p, q])*(t1[x][y]*t2[x+p][y+q]/(p*p + q*q))
                vector = vector + np.array([p, q])*(t[x+p][y+q]/(p*p + q*q))
    return vector


u = []
v = []
mv = []
if(len(x_grid) - W > W and len(y_grid) - W > W):
    for i in range(W, len(y_grid) - W):
        x_tr = []
        y_tr = []
        h = []
        for j in range(W, len(x_grid) - W):
            result = FMap(j, i)
            x_tr.append(result[0])
            y_tr.append(result[1])
            h.append([result[0], result[1]])
        u.append(x_tr)
        v.append(y_tr)
        mv.append(h)
    U = np.asarray(u)
    V = np.asarray(v)
    [X, Y] = np.meshgrid(x_grid[W: len(x_grid)-W], y_grid[W: len(y_grid)-W])
    
fig, ax = plt.subplots(figsize=(15,7))
img = plt.imread("pr.JPG")
ax.imshow(img, extent = ext)
strm = ax.streamplot(X/cell, Y/cell, U/cell, V/cell, color = np.log(np.sqrt(U**2 + V**2)), density = 1, linewidth=1.5, cmap=plt.cm.OrRd)
plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.3)
plt.colorbar(strm.lines, cax=cax)
#fig0.colorbar(strm.lines)
ax.set_title('Flow Map of Puerto Rico (08/28/2017 7am to 10am)')
plt.show()

