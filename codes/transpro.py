

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('sol.csv',nrows=100,engine='python')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
    
corpus = []

for i in range(0, 100):
    rev=re.sub(r'http\S+', '',data['text'][i] )
    review = re.sub('[^a-zA-Z0-9]', ' ',rev)
    review = review.lower()
    corpus.append(review)
    
data['text']=corpus

from googletrans import Translator
translator = Translator()

                        
#Diff code
translation=[]
X=data.iloc[:,0].values

X=X.tolist()
for i in range(1000,1100):
      translation.append(translator.translate(X[i]))

#save to dataframe
data['text']=translation

#save to xlxs


data.to_excel("tp.xlsx")
    
    
    
    
    
