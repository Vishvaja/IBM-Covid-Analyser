
#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

data=pd.read_csv('Book2.csv',engine='python' )

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 499):
    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    sw=stopwords.words('english')
    sw.remove('not')
    review = [ps.stem(word) for word in review if not word in set(sw) ]
    review = ' '.join(review)
    corpus.append(review)
import csv
with open('Book2.csv','w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(corpus)


corpus=np.array(corpus)

import nltk
nltk.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=corpus.apply(lambda x:sia.polarity_scores(x)['compound'])

#Sample result
data.head()
