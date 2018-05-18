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


def kdemap(t_start, t_end):
    values = data[(data['utc'] >= t_start) & (data['utc'] < t_end)].as_matrix(["x", "y"]).T
#    print len(data[(data['utc'] >= t_start) & (data['utc'] < t_end)].as_matrix(["x", "y"]))
    kde = stats.gaussian_kde(values)
    prob = kde.evaluate([np.meshgrid(x_grid, y_grid)[0].T.ravel(), np.meshgrid(x_grid, y_grid)[1].T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell *cell

    period = time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_start)) + '-' + time.strftime('%m/%d/%Y %H:%M:%S',  time.localtime(t_end)).split(' ')[1]
    title = 'Kernel Density Estimation\n' + period
    name = 'figures/kde' + str(t_start) + '-' + str(t_end) + '-' + str(cell) + '.png'
    DrawKDE(title, name, prob)
    print "Finish getting probability"
    
    return prob

def DrawKDE(title, name, prob):
    fig, ax = plt.subplots(figsize=(10,15))
    img = plt.imread("wl.JPG")
    ax.imshow(img, extent = ext)
    im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
    plt.xticks(x_range, x_label)
    plt.yticks(y_range, y_label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    ax.set_title(title)
    plt.colorbar(im, cax=cax)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font)
#    fig.savefig(name)   # save the figure to file
    print name + " saved..."
    plt.show()

def DrawFlow(title, name, U, V):
    global X, Y, mask
    X, Y = np.meshgrid(x_grid, y_grid)
    fig, ax = plt.subplots(figsize=(10,15))
    img = plt.imread("wl.JPG")
    ax.imshow(img, extent = ext)
#    mask = (np.round(10**12*np.sqrt(U**2 + V**2))>1).astype(int)
#    U = np.multiply(mask, U)
#    V = np.multiply(mask, V)
    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.log2(np.sqrt(U**2 + V**2)), density = 0.5, linewidth=1.2, cmap=plt.cm.Purples, arrowsize=1.2)
#    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.round(2*10**10*np.sqrt(U**2 + V**2)), density = 0.5, linewidth=np.round(2*10**10*np.sqrt(U**2 + V**2)), cmap=plt.cm.Purples, arrowsize=0.5)
#    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = 'k', density = 0.5, linewidth=np.round(2*10**3*np.sqrt(U**2 + V**2)), arrowsize=0.5)
    ax.set_ylim([y_min/cell, y_max/cell])
    ax.set_xlim([x_min/cell, x_max/cell])
    plt.xticks(x_range, x_label)
    plt.yticks(y_range, y_label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    plt.colorbar(strm.lines, cax=cax)
    #fig0.colorbar(strm.lines)
    ax.set_title(title)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font)
#    fig.savefig(name)
    print name + " saved..."
    plt.show()

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
    title = alg + " Based Flow Map of Puerto Rico \n" + time.strftime('%m/%d/%Y %H%p',  time.localtime(t1)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime(t2)).split(' ')[1] 
    DrawFlow(title, name, U, V)
    
    
t1_1 = 1394721000
t1_2 = t1_1 + 3600
t2_1 = 1394728200
t2_2 = t2_1 + 3600
prob1 = kdemap(t1_1, t1_2)
prob2 = kdemap(t2_1, t2_2)

name = "figures/diff" + str((t1_1 + t1_2)/2) + "-" + str((t2_1 + t2_2)/2) + "-" + str(cell) + ".png"
title = "Difference of Two Timestamps\n" + time.strftime('%m/%d/%Y %H%p',  time.localtime((t1_1 + t1_2)/2)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime((t2_1 + t2_2)/2)).split(' ')[1]
DrawKDE(title, name, prob2 - prob1)

#FlowMap(prob1, prob2, "Gradient", (t1_1 + t1_2)/2, (t2_1 + t2_2)/2)
FlowMap(prob1, prob2, "Gravity", (t1_1 + t1_2)/2, (t2_1 + t2_2)/2)

#for i in range(24):
#    print len()

#title = "Vector Field(Gravity Based Model)\n" + time.strftime('%m/%d/%Y %H%p',  time.localtime((t1_1 + t1_2)/2)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime((t2_1 + t2_2)/2)).split(' ')[1]
#fig, ax = plt.subplots(figsize=(7,15))
#img = plt.imread("wl.JPG")
#ax.imshow(img, extent = ext)
#plt.quiver(X/(cell*1.0), Y/(cell*1.0), U, V, units='width')
#plt.xticks(x_range, x_label)
#plt.yticks(y_range, y_label)
#plt.xticks(x_range, x_label)
#plt.yticks(y_range, y_label)
#ax.set_title(title)
##qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
#plt.show()

#global X, Y, mask
#X, Y = np.meshgrid(x_grid, y_grid)
#fig, ax = plt.subplots(figsize=(10,15))
#img = plt.imread("wl.JPG")
#ax.imshow(img, extent = ext)
##mask = (np.round(10**12*np.sqrt(U**2 + V**2))>1).astype(int)
##U = np.multiply(mask, U)
##V = np.multiply(mask, V)
#strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.sqrt(U**2 + V**2), density = 0.5, linewidth=1, cmap=plt.cm.Purples, arrowsize=0.5)
##    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.round(2*10**10*np.sqrt(U**2 + V**2)), density = 0.5, linewidth=np.round(2*10**10*np.sqrt(U**2 + V**2)), cmap=plt.cm.Purples, arrowsize=0.5)
#ax.set_ylim([y_min/cell, y_max/cell])
#ax.set_xlim([x_min/cell, x_max/cell])
#plt.xticks(x_range, x_label)
#plt.yticks(y_range, y_label)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="2%", pad=0.3)
#plt.colorbar(strm.lines, cax=cax)
##fig0.colorbar(strm.lines)
#ax.set_title(title)
#plt.show()
#fig.savefig(name)
#print name + " saved..."

#title = "Vector Length (Graient Based Model)\n" + time.strftime('%m/%d/%Y %H%p',  time.localtime((t1_1 + t1_2)/2)) + ' to ' + time.strftime('%m/%d/%Y %H%p',  time.localtime((t2_1 + t2_2)/2)).split(' ')[1]
#fig, ax = plt.subplots(figsize=(10,15))
#img = plt.imread("wl.JPG")
#ax.imshow(img, extent = ext)

#im = ax.imshow(np.rot90(np.sqrt(v**2 + u**2).T), cmap=plt.cm.gist_earth_r, alpha=0.8, extent = ext)
#plt.quiver(X/(cell*1.0), Y/(cell*1.0), U, V, units='width', alpha=0.1)
#plt.xticks(x_range, x_label)
#plt.yticks(y_range, y_label)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="2%", pad=0.3)
#ax.set_title(title)
#plt.colorbar(im, cax=cax)
#plt.show()
