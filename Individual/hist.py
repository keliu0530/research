#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

# data = pd.read_csv("user.csv")
# x = data["COUNT(*)"].as_matrix()
# del data

# for i in range(len(x)):
# 	if(x[i] > 6000):
# 		x[i] = 6000

# b = [0, 10, 50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
# n, bins, patches = plt.hist(x, b, facecolor='g', alpha=0.75)


# plt.xlabel('Number of record')
# plt.ylabel('Number of user')
# plt.title('Histogram ($\mu=389.1,\ \sigma=1127.9$)')
# # plt.text(0, 6000, r'$\mu=100,\ \sigma=15$')
# plt.axis([0, 6500, 0, 40000])
# plt.grid(True)
# plt.show()

user = pd.read_csv("user1.csv").sort_values(['utc'])

record = user.utc.values

record = np.insert(record, 0, user.iloc[0].utc)

record = np.delete(record, len(record) - 1)

interval = (user.utc.values - record)

# interval = [0]

# for i in range(1, len(user)):
# 	interval.append(user.iloc[i].utc - user.iloc[i - 1].utc)

# for i in range(len(x)):
# 	if(x[i] > 6000):
# 		x[i] = 6000

# b = [0, 10, 50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
# n, bins, patches = plt.hist(x, b, facecolor='g', alpha=0.75)
plt.figure(figsize=(15,7))
plt.plot(range(len(interval)), interval)
# plt.xlabel('hr')
plt.ylabel('hr')
plt.title('Interval')
# plt.text(0, 6000, r'$\mu=100,\ \sigma=15$')
# plt.axis([0, 6500, 0, 40000])
# plt.grid(True)
plt.show()