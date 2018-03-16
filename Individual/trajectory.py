import pandas as pd
import matplotlib.pyplot as plt

user = pd.read_csv("user1.csv")

user = user.sort_values(by=['utc'])

x = user['x'].values.tolist()

y = user['y'].values.tolist()

plt.plot(x,y)