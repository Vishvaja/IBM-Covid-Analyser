import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

data=pd.read_csv('Verified.csv',engine='python')
X=data.iloc[:,0].values
vcount=0
ncount=0
for i in range(463418):
    if(X[i]==1):
        vcount+=1
    else:
        ncount+=1
        
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(463418):
    rev=re.sub(r'http\S+', '',data['text'][i] )
    review = re.sub('[^a-zA-Z0-9]', ' ',rev)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    sw=stopwords.words('english')
    sw.remove('not')
    review = [ps.stem(word) for word in review if not word in set(sw) ]
    review = ' '.join(review)
    corpus.append(review)


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])

v=[]
n=[]
for i in range(463418):
    if(X[i]==1):
        v.append(data['compound'][i])
    else:
        n.append(data['compound'][i])
vlist=[0,0,0]  
nlist=[0,0,0] 
for i in range(vcount):
    if(v[i]>0):
        vlist[0]+=1
    elif(v[i]==0):
        vlist[1]+=1
    else:
        vlist[2]+=1
for i in range(ncount):
    if(n[i]>0):
        nlist[0]+=1
    elif(n[i]==0):
        nlist[1]+=1
    else:
        nlist[2]+=1
 
#Pie Chart for Verified
labels = 'Positive','Neutral','Negative'
sizes=vlist
explode = (0.1, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

fig1.savefig('F:/Machine-Learning-A-Z-Template-Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/plots/verified.png')

#Pie Chart for Non-Verified    
labels = 'Positive','Neutral','Negative'
sizes=nlist
explode = (0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig2, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
fig2.savefig('F:/Machine-Learning-A-Z-Template-Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/plots/nonverified.png')

