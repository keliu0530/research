import pandas as pd
import matplotlib.pyplot as plt
user = pd.read_csv("user1.csv") 
user = user.sort(['utc'])

x = user['x'].values.tolist()

y = user['y'].values.tolist()

plt.plot(x,y)

plt.show()

interval = []

for i in range(len(user)):
	if(i > 0):
		if(user.iloc[i]['utc'] - user.iloc[i - 1]['utc'] > 3600): print user.iloc[i]['utc'] 						
		interval.append(user.iloc[i]['utc'] - user.iloc[i - 1]['utc'])
