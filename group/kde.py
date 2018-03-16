#!/usr/bin/python
import MySQLdb
from os import listdir
import pandas as pd
import utm
import numpy as np
from pyproj import Proj, transform
import os.path
import scipy.stats as stats

db = MySQLdb.connect(host="keliudb.corzaau5yusv.us-east-2.rds.amazonaws.com",    # your host, usually localhost
                     user="keliu",         # your username
                     passwd="19930530",  # your password
                     db="PuertoRico")        # name of the data base

df = pd.read_sql("SELECT x, y, utc FROM PhoneData WHERE utc >= 1503892800 and utc < 1503979200", con=db)

values = df.as_matrix(["x", "y"]).T

del df

#kde = stats.gaussian_kde(values)

#del values

#x = range(112000, 288000, 100)
#y = range(210000, 275000, 100)

#kde.evaluate([np.meshgrid(x,y)[0].T.ravel(), np.meshgrid(x,y)[1].T.ravel()]).reshape(len(x), len(y))
