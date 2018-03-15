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
    outProj = Proj(init='epsg:4437')
    x, y = transform(inProj, outProj, latlon[0], latlon[1])
    return [x, y]
    
db = MySQLdb.connect(host="keliudb.corzaau5yusv.us-east-2.rds.amazonaws.com",    # your host, usually localhost
                     user="keliu",         # your username
                     passwd="19930530",  # your password
                     db="PuertoRico")        # name of the data base

temp = pd.read_sql("SELECT * FROM PuertoRico WHERE safegraph_id = '2b261110b878dd50de54740c91273643aa8ffaa7f804324556f11f0ce1a5f518';", con=db)

temp['x'] = None
temp['y'] = None

for i in range(len(temp)):
    record = [temp.iloc[i]['longitude'], temp.iloc[i]['latitude']]
    result = transfer(record)
    if(i%1000 == 0): print i/1000
    temp.set_value(i, 'x', result[0])
    temp.set_value(i, 'y', result[1])
    del result
    del record
