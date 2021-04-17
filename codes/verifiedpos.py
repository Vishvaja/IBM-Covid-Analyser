import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

data=pd.read_csv('Verified.csv',engine='python')
X=data.iloc[:,0].values
vcount=0
ncount=0
for i in range(4631):
    if(X[i]==1):
        vcount+=1
    else:
        ncount+=1
        
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(4631):
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

data=data.iloc[:4631,:]
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])

X=data.iloc[:,0].values
Y=data.iloc[:,5].values

ver=[0,0,0]
nver=[0,0,0]
for i in range(4631):
    if(X[i]==1):
        if(Y[i]>0):
            ver[0]=ver[0]+1
        elif(Y[i]<0):
            ver[1]=ver[1]+1
        else:
            ver[2]=ver[2]+1
    else:
        if(Y[i]>0):
            nver[0]=nver[0]+1
        elif(Y[i]<0):
            nver[1]=nver[1]+1
        else:
            nver[2]=nver[2]+1
            
D=[ver,nver]
 
Z=[]
for i in range(len(D[0])): 
        # print(i) 
    row =[] 
    for item in D: 
         row.append(item[i]) 
    Z.append(row) 
    
    
X = np.arange(3)
#fig = plt.figure()
fig,ax=plt.subplots()
#ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, D[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, D[1], color = 'r', width = 0.25)
ax.legend(labels=['Verified', 'Non-Verified'])
plt.xlabel('Sentiment scores')
plt.ylabel('No.of Verified and Non verified users')
#ax.bar(X + 0.50, Z[2], color = 'r', width = 0.25)
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/pnn.png")
  
X = np.arange(2)
fig = plt.figure()
fig,ax=plt.subplots()
#ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, Z[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, Z[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, Z[2], color = 'r', width = 0.25)
ax.legend(labels=['Positive', 'Negative','Neutral'])
plt.xlabel('Sentiment scores')
plt.ylabel('No.of Verified and Non verified users')
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/nv.png")
  


