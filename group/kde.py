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

x_min = 112000
x_max = 288000
y_min = 210000
y_max = 275000

global x_grid, y_grid, W, cell, x_label, y_label, mask_U, mask_V, dist, font
font = 20
cell = 100
x_grid = range(x_min + cell/2, x_max + cell/2, cell)
y_grid = range(y_min + cell/2, y_max + cell/2, cell)
ext = [x_min/cell, x_max/cell, y_min/cell, y_max/cell]
W = 2
x_label = ['120km', '140km', '160km', '180km', '200km', '220km', '240km', '260km', '280km']
y_label = ['220km', '240km', '260km', '280km']

temp = []
for i in range(-W, W+1):
    temp.append(-1 * i * np.ones(W * 2 + 1))
mask_V = np.array(temp)
del temp
mask_U = mask_V.T
#mask_U = - mask_U
dist = mask_U**2 + mask_V**2
dist[W][W] = 1



def kdemap(t_start, t_end):
    if("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy" in os.listdir("./")):
        prob = np.load("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy")
    else:
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
    
        

        kde = stats.gaussian_kde(values)
        prob = kde.evaluate([np.meshgrid(x_grid, y_grid)[0].T.ravel(), np.meshgrid(x_grid, y_grid)[1].T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell *cell
        np.save("prob-" + str(t_start) + "-" + str(t_end) + "-" + str(cell) + ".npy", prob)
        print "Finish calculating kde..."

    period = time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_start)) + '-' + time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_end)).split(' ')[1]
    title = 'Kernel Density Estimation(' + period + ')'
    name = 'figures/kde' + str(t_start) + '-' + str(t_end) + '-' + str(cell) + '.png'
    DrawKDE(title, name, prob)
    print "Finish getting probability"
    
    return prob

def DrawKDE(title, name, prob):
    fig, ax = plt.subplots(figsize=(15,7))
    img = plt.imread("pr.JPG")
    ax.imshow(img, extent = ext)
    im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    ax.set_title(title)
    plt.colorbar(im, cax=cax)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font)
    fig.savefig(name)   # save the figure to file
    print name + " saved..."
    
#def DrawDiff(title, name, prob):
#    X, Y = np.meshgrid(x_grid, y_grid)
#    fig, ax = plt.subplots(figsize=(15,7))
#    img = plt.imread("pr.JPG")
#    ax.imshow(img, extent = ext)
#    im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
#    plt.contour(X/cell, Y/cell, prob.T, colors='black');
#    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
#    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="2%", pad=0.3)
#    ax.set_title(title)
#    plt.colorbar(im, cax=cax)

#    plt.show()
#    fig.savefig(name)   # save the figure to file
#    print name + " saved..."


#def Vec(x, y, t):
#    t = t2 - t1
#    vector = np.zeros(2)
#    for p in range(-W, W+1):
#        for q in range(-W, W+1):
#            if(sqrt(p*p + q*q) != 0):
#                vector = vector + np.array([p, q])*(t1[x][y]*t2[x+p][y+q]/(p*p + q*q))
##                vector = vector + np.array([p, q])*(t1[x][y]**2*t[x+p][y+q]**3/(p*p + q*q))
#    return vector
    
#def gravity(t1, t2):
#    U = np.multiply(t1, signal.convolve2d(t2, mask_U/dist, boundary='wrap', mode='same')).T
#    V = np.multiply(t1, signal.convolve2d(t2, mask_V/dist, boundary='fill', fillvalue = 0, mode='same')).T
#    return U, V
#DrawFlow(U,V)



#tt = np.multiply(t1, con)
#result = signal.convolve2d(tt, mask_U, 'valid')

#def Vec(x, y, t):
##    t = t2 - t1
#    vector = np.zeros(2)
#    for p in range(-W, W+1):
#        for q in range(-W, W+1):
#            loc_x = x + p
#            loc_y = y + q
#            if(loc_x < 0): loc_x = 0
#            if(loc_x >= len(t)): loc_x = len(t) - 1
#            if(loc_y < 0): loc_y = 0
#            if(loc_y >= len(t[0])): loc_y = len(t[0]) - 1
#            if(sqrt(p*p + q*q) != 0):
#                 vector = vector + np.array([p, q])*(t1[x][y]*t2[loc_x][loc_y]/(p*p + q*q))
##                vector = vector + np.array([p, q])*(t1[x][y]*t2[x+p][y+q]/(p*p + q*q))
##                vector = vector + np.array([p, q])*(t1[x][y]**2*t[loc_x][loc_y]/(p*p + q*q))
#    return vector
    
def DrawFlow(title, name, U, V):
    X, Y = np.meshgrid(x_grid, y_grid)
    fig, ax = plt.subplots(figsize=(15,7))
    img = plt.imread("pr.JPG")
    ax.imshow(img, extent = ext)
    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.log2(np.sqrt(U**2 + V**2)), density = 1.2, linewidth=1, cmap=plt.cm.OrRd, arrowsize=0.7)
    ax.set_ylim([y_min/cell, y_max/cell])
    ax.set_xlim([x_min/cell, x_max/cell])
    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    plt.colorbar(strm.lines, cax=cax)
    #fig0.colorbar(strm.lines)
    ax.set_title(title)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font)
    fig.savefig(name)
    print name + " saved..."
#    plt.show()
    
    
#def Grad(prob1, prob2):
#    V, U = np.gradient(prob2.T - prob1.T)
#    X, Y = np.meshgrid(x_grid, y_grid)
#    fig, ax = plt.subplots(figsize=(15,7))
#    img = plt.imread("pr.JPG")
#    ax.imshow(img, extent = ext)
#    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.log(np.sqrt(U**2 + V**2)), density = 1.2, linewidth=1, cmap=plt.cm.OrRd, arrowsize=0.7)
#    plt.contour(X/cell, Y/cell, (prob2 - prob1).T, 10, colors='black');
#    ax.set_ylim([y_min/cell, y_max/cell])
#    ax.set_xlim([x_min/cell, x_max/cell])
#    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
#    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="2%", pad=0.3)
#    plt.colorbar(strm.lines, cax=cax)
#    #fig0.colorbar(strm.lines)
#    plt.show()
#    
#Grad(prob1, prob2)

#def Grav(prob1, prob2):
#    U = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_U/dist, boundary='fill', fillvalue = 0, mode='same'))
#    V = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_V/dist, boundary='fill', fillvalue = 0, mode='same'))
#        
#    X, Y = np.meshgrid(x_grid, y_grid)
#    fig, ax = plt.subplots(figsize=(15,7))
#    img = plt.imread("pr.JPG")
#    ax.imshow(img, extent = ext)
#    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.log(np.sqrt(U**2 + V**2)), density = 1.2, linewidth=1, cmap=plt.cm.OrRd, arrowsize=0.7)
#    ax.set_ylim([y_min/cell, y_max/cell])
#    ax.set_xlim([x_min/cell, x_max/cell])
#    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
#    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="2%", pad=0.3)
#    plt.colorbar(strm.lines, cax=cax)
#    #fig0.colorbar(strm.lines)
#    plt.show()
#    
#Grav(prob1, prob2)

#def FlowMap(prob1, prob2, alg, t1, t2):
#    global U, V
#    if(alg == "Gradient"):
#        U, V = np.gradient(prob2 - prob1)
#        U = U.T
#        V = V.T
#    elif(alg == "Gravity"):
#        U = np.multiply(prob1, signal.convolve2d(prob2, mask_U/dist, boundary='fill', fillvalue = 0, mode='same')).T
#        V = np.multiply(prob1, signal.convolve2d(prob2, mask_V/dist, boundary='fill', fillvalue = 0, mode='same')).T
#        
#    name = 'figures/' + alg + 'Model' + str(t1) + "-" + str(t2) + "-" +str(cell)
#    title = alg + " Based Flow Map of Puerto Rico (" + time.strftime('%m/%d/%Y %H%p',  time.localtime(t1)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime(t2)).split(' ')[1] + ")"
#    DrawFlow(title, name, U, V)

def FlowMap(prob1, prob2, alg, t1, t2):
    global U, V
    if(alg == "Gradient"):
        V, U = np.gradient(prob2.T - prob1.T)
#        U = U.T
#        V = V.T
    elif(alg == "Gravity"):
        U = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_U/dist, boundary='fill', fillvalue = 0, mode='same'))
        V = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_V/dist, boundary='fill', fillvalue = 0, mode='same'))
        
    name = 'figures/' + alg + 'Model' + str(t1) + "-" + str(t2) + "-" +str(cell)
    title = alg + " Based Flow Map of Puerto Rico (" + time.strftime('%m/%d/%Y %H%p',  time.localtime(t1)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime(t2)).split(' ')[1] + ")"
    DrawFlow(title, name, U, V)



#    return U, V
#        return U, V
#    elif(alg == "graivity"):
#        u = []
#        v = []
#        mv = []
#    #    if(len(x_grid) - W > W and len(y_grid) - W > W):
#    #        for i in range(W, len(y_grid) - W):
#        for i in range(len(y_grid)):
#            x_tr = []
#            y_tr = []
#            h = []
#    #            for j in range(W, len(x_grid) - W):
#            for j in range(len(x_grid)):
#                result = Vec(j, i, t2 - t1)
#                x_tr.append(result[0])
#                y_tr.append(result[1])
#                h.append([result[0], result[1]])
#            u.append(x_tr)
#            v.append(y_tr)
#            mv.append(h)
#        U = np.asarray(u)
#        V = np.asarray(v)
#    return U, V
#    [X, Y] = np.meshgrid(x_grid[W: len(x_grid)-W], y_grid[W: len(y_grid)-W])
    




#FlowMap(prob1, prob2, "gradient")
#DrawFlow()

#8.28.2017 6:30am to 7:30am
t1_1 = 1503916200
t1_2 = 1503919800
#8.28.2017 9:30am to 10:30am
t2_1 = 1503927000
t2_2 = 1503930600

##8.28.2017 3:30Pm to 4:30pm
#t1_1 = 1503948600
#t1_2 = 1503952200
##8.28.2017 6:30pm to 7:30pm
#t2_1 = 1503959400
#t2_2 = 1503963000

##8.28.2017 1:30am to 2:30pm
#t1_1 = 1503898200
#t1_2 = 1503901800
##8.28.2017 4:30pm to 5:30pm
#t2_1 = 1503909000
#t2_2 = 1503912600

prob1 = kdemap(t1_1, t1_2)
prob2 = kdemap(t2_1, t2_2)
name = "figures/diff" + str((t1_1 + t1_2)/2) + "-" + str((t2_1 + t2_2)/2) + "-" + str(cell) + ".png"
title = "Difference of Two Timestamps (" + time.strftime('%m/%d/%Y %H%p',  time.localtime((t1_1 + t1_2)/2)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime((t2_1 + t2_2)/2)).split(' ')[1] + ')'
DrawKDE(title, name, prob2 - prob1)

FlowMap(prob1, prob2, "Gravity", (t1_1 + t1_2)/2, (t2_1 + t2_2)/2)
FlowMap(prob1, prob2, "Gradient", (t1_1 + t1_2)/2, (t2_1 + t2_2)/2)
#time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime((t_1 + t_2)/2))
#time.strftime('%m/%d/%Y %H%p',  time.localtime((t1_1 + t1_2)/2)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime((t2_1 + t2_2)/2)).split(' ')[1]

