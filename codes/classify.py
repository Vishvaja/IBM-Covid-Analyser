#Importing Libraries

import pandas as pd
import numpy as np
#Importing Dataset
dataset=pd.read_csv('legend.csv',engine="python")
x=dataset.iloc[:,1].values
#Importing Libraries


i=10001

while i<100023:
    y = dataset.iloc[:i,3].values
    dataset=pd.read_csv('output3.csv',nrows=i+200,engine="python")

    corpus=x
    

# Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    X_train=[]
    X_test=[]
    for j in range (i+200):
        if j>=i and j<i+200:
            X_test.append(X[j])
        else:
            X_train.append(X[j])
    #BernoulliNB
    from sklearn.naive_bayes import BernoulliNB
    classifier = BernoulliNB()
    classifier.fit(X_train, y)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y=y.tolist()
    for k in range(200):
        y.append(y_pred[k])
    dataset['new']=pd.Series(y)
    i+=200
    
    
def isNaN(string):
    return string != string

for i in range(100023):
    if(isNaN(x[i])):
        x[i]="coronavirus"
from openpyxl.workbook import Workbook
dataset.to_excel("legend.xlsx")