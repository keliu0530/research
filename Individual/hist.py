#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("user.csv")
x = data["COUNT(*)"].as_matrix()
del data

for i in range(len(x)):
	if(x[i] > 6000):
		x[i] = 6000

# plt.hist(x, )  # plt.hist passes it's arguments to np.histogram

# plt.hist(x, normed=True, bins=100)
# plt.show()

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)

# the histogram of the data
b = [0, 10, 50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
n, bins, patches = plt.hist(x, b, facecolor='g', alpha=0.75)


plt.xlabel('Number of record')
plt.ylabel('Number of user')
plt.title('Histogram ($\mu=389.1,\ \sigma=1127.9$)')
# plt.text(0, 6000, r'$\mu=100,\ \sigma=15$')
plt.axis([0, 6500, 0, 40000])
plt.grid(True)
plt.show()

# mu, sigma = np.mean(x), np.std(x)
# # x = mu + sigma*np.random.randn(10000)

# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)

# # add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)

# plt.show()