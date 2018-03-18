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

x_min = 112000
x_max = 288000
y_min = 210000
y_max = 275000
cell = 1000

global x_grid, y_grid, W, T
x_grid = range(x_min + cell/2, x_max + cell/2, cell)
y_grid = range(y_min + cell/2, y_max + cell/2, cell)
ext = [x_min/1000, x_max/1000, y_min/1000, y_max/1000]
W = 2
T = 1


def kdemap(t_start, t_end):
    if(str(t_start) + "-" + str(t_end) + ".npy" in os.listdir("./")):
        values = np.load(str(t_start) + "-" + str(t_end) + ".npy")
#        return values
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
    fig.savefig('figures/kde' + str(t_start) + '-' + str(t_end) + '.png')   # save the figure to file


#kdemap(1503916200, 1503919800)
kdemap(1503927000, 1503930600)

#def FMap(kde, x, y):
#    xyt = np.zeros(3)
#    t = len(kde)/2
#    ft = kde[t][x][y]
#    for p in range(-W, W+1):
#        for q in range(-W, W+1):
#            for r in range(-T, T+1):
#                ftr = kde[t+r][x+p][y+q]
#                if sqrt(p*p + q*q) != 0:
#                    xyt = xyt + np.array([p, q, r])*ft*ftr/sqrt(p*p + q*q)
#    return xyt*(10**22)



#u = []
#v = []
#mv = []
#if(len(x_grid) - W > W and len(y_grid) - W > W):
#    for i in range(W, len(y) - W):
#        x_tr = []
#        y_tr = []
#        h = []
#        for j in range(W, len(x) - W):
#            result = FMap(kde, j, i)
#            x_tr.append(result[0])
#            y_tr.append(result[1])
#            h.append([result[0], result[1]])
#        u.append(x_tr)
#        v.append(y_tr)
#        mv.append(h)
#    U = np.asarray(u)
#    V = np.asarray(v)
#    [X, Y] = np.meshgrid(x[W: len(x)-W], y[W: len(y)-W])
#plt.show()

