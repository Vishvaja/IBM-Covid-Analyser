# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:22:31 2020

@author: pjai1
"""

from flask import Flask ,render_template
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
    data=pd.read_excel('output1.xlsx')
    x=data.iloc[:,:].values
    y=len(x)
    return render_template("index.html",variable=y)

'''@app.route('/display')
def display():
    
    return y'''

if __name__ =='__main__':
    app.run()