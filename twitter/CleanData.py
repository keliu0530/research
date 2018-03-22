#!/usr/bin/python
import MySQLdb
from os import listdir
import pandas as pd
import utm
import numpy as np
from pyproj import Proj, transform
import os.path

def transfer(latlon):
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:26974')
    x, y = transform(inProj, outProj, latlon[0], latlon[1])
    return [x, y]
    
    
data = pd.read_csv("West Lafayette 2014.csv")
df = data[(data['epoch'] >= 1394681400) & (data['epoch'] < 1394771400)][["epoch", "longitude", "latitude"]]
del data

x = []
y = []
t = []

for i in range(len(df)):
    result = transfer(df.iloc[i][["longitude", "latitude"]])
    x.append(result[0])
    y.append(result[1])
    t.append(df.iloc[i]["epoch"])

data = pd.DataFrame(columns = ['x', 'y', 'utc'])
data['utc'] = t
data['x'] = x
data['y'] = y
