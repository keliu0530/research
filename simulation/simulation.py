import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from bresenham import bresenham
from math import sqrt 

global cell, rg, font, x_grid, y_grid, ext, X, Y
cell = 1
rg = 100
font = 20

x_grid = range(0 + cell/2, rg + cell/2, cell)
y_grid = range(0 + cell/2, rg + cell/2, cell)
ext = [0, rg/cell, 0, rg/cell]
X, Y = np.meshgrid(x_grid, y_grid)

def kdemap(values):
    kde = stats.gaussian_kde(values)
    prob = kde.evaluate([X.T.ravel(), Y.T.ravel()]).reshape(len(x_grid), len(y_grid)) * cell *cell
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(np.rot90(prob), cmap=plt.cm.gist_earth_r, alpha=0.7, extent = ext)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.3)
    plt.colorbar(im, cax=cax)
    #for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    #    item.set_fontsize(font)
    plt.show()
    return prob
    
    

mean = [30, 40]
cov = [[100, 0], [0, 100]]  # diagonal covariance
x1, y1 = np.random.multivariate_normal(mean, cov, 1000).T

#mean = [60, 70]
#cov = [[50, 0], [0, 50]]  # diagonal covariance
#x2, y2 = np.random.multivariate_normal(mean, cov, 1000).T

x2 = x1 + 30
y2 = y1 +30

plt.figure(figsize=(7,7))
plt.plot(x1, y1, 'r.')
plt.plot(x2, y2, 'b.')
plt.xlim(0, rg)
plt.ylim(0, rg)
plt.show()

prob1 = kdemap(np.stack((x1, y1)))
prob2 = kdemap(np.stack((x2, y2)))


def gravity(prob1, prob2, W):
    temp = []
    for i in range(-W, W+1):
        temp.append(-1 * i * np.ones(W * 2 + 1))
    mask_V = np.array(temp)
    del temp
    mask_U = mask_V.T
    #mask_U = - mask_U
    dist = mask_U**2 + mask_V**2
    dist[W][W] = 1
    U = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_U/dist, boundary='fill', fillvalue = 0, mode='same'))
    V = np.multiply(prob1.T, signal.convolve2d(prob2.T, mask_V/dist, boundary='fill', fillvalue = 0, mode='same'))
    return U, V
    
def gradient(prob1, prob2):
    V, U = np.gradient(prob2.T - prob1.T)
    return U, V

def vec(x1, y1, x2, y2):
    U = np.zeros((rg/cell, rg/cell))
    V = np.zeros((rg/cell, rg/cell))
    for i in range(len(x1)):
        for j in bresenham(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])):
            dist = sqrt((x2[i] - x1[i])**2 + (y2[i] - y1[i])**2)
            if(j[0] < 100 and j[0] >= 0 and j[1] < 100 and j[1] >= 0):
                U[j[0]][j[1]] += (x2[i] - x1[i])/dist
                V[j[0]][j[1]] += (y2[i] - y1[i])/dist
    return U, V

def glyph(U, V):
    plt.figure()
    Q = plt.quiver(X, Y, U, V, units='width')
    plt.show()

U1, V1 = gravity(prob1, prob2, 100)
glyph(U1, V1)
#U, V = gravity(prob1, prob2, 100)
#U, V = gravity(prob2, prob1, 100)
#U, V = gradient(prob1, prob2)

#U = U1 * 420929338 - U2
#V = V1 * 420929338 - V2
#U, V = vec(x1, y1, x2, y2)
#U, V = vec(x2, y2, x1, y1)

U2, V2 = vec(x1, y1, x2, y2)
glyph(U2, V2)

U = U2
U[U == 0] = 1
M = (np.multiply(U1, U2) + np.multiply(V1, V2))/np.sqrt(np.multiply(np.multiply(U1, U1) + np.multiply(V1, V1), np.multiply(U, U) + np.multiply(V1, V2)))
#FlowMap(prob1, prob2, "Gradient")
#    DrawFlow(title, name, U, V)

#def DrawFlow(title, name, U, V):
#    X, Y = np.meshgrid(x_grid, y_grid)
#    fig, ax = plt.subplots(figsize=(15,7))
#    img = plt.imread("pr.JPG")
#    ax.imshow(img, extent = ext)
#    strm = ax.streamplot(X/(cell*1.0), Y/(cell*1.0), U/(cell*1.0), V/(cell*1.0), color = np.log2(np.sqrt(U**2 + V**2)), density = 1.2, linewidth=1, cmap=plt.cm.OrRd, arrowsize=0.7)
#    ax.set_ylim([y_min/cell, y_max/cell])
#    ax.set_xlim([x_min/cell, x_max/cell])
#    plt.xticks(range(120000/cell, 300000/cell, 20000/cell), x_label)
#    plt.yticks(range(220000/cell, 280000/cell, 20000/cell), y_label)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="2%", pad=0.3)
#    plt.colorbar(strm.lines, cax=cax)
#    #fig0.colorbar(strm.lines)
#    ax.set_title(title)
#    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#        item.set_fontsize(font)
#    fig.savefig(name)
#    print name + " saved..."
