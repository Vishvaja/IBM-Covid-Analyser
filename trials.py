# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:30:09 2020

@author: pjai1
"""

import twint
import nest_asyncio
nest_asyncio.apply()
import pandas as pd

csvfile=pd.read_csv('tweets.csv',engine="python")
X=csvfile.iloc[:,7].values
#print([x[0] for x in csv.reader(csvfile)])

#userx = [x[0] for x in csv.reader(csvfile)]
c = twint.Config()
n=len(csvfile)
c.Username = 'stephenfoley'
if (c.Verified==True):
    print("Yay")
else:
    print("False")
twint.run.Search(c)
    
for i in range (188,n):
    c.Username = csvfile['username'][i]
    c.Search="coronavirus"
    c.Since="2020-06-26 10:46:45"
    c.Until="2020-06-27 00:00:00"
    if c.Verified==True:
        csvfile['geo'][i]=1
    else:
        csvfile['geo'][i]=0
    twint.run.Search(c)


      
            