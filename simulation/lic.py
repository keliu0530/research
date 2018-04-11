import numpy as np
import matplotlib.pyplot as plt

global cell, rg, font, x_grid, y_grid, ext, X, Y
cell = 1
rg = 10

x_grid = range(0, rg, cell)
y_grid = range(0, rg, cell)
ext = [0, rg/cell, 0, rg/cell]
X, Y = np.meshgrid(x_grid, y_grid)
#Y = np.flip(Y,0)
U = -2 * (X - rg/3)
V = 2 * (Y - rg/3)

plt.figure()
plt.quiver(X, Y, U, V, units='width')
plt.show()

vec = np.asarray(np.stack((U, V), axis=-1)) 

x0 = 5
y0 = 5

line_x = [x0]
line_y = [y0]

P0 = np.array(x0, y0)
P1 = P0 + 
