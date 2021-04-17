import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('Posneg.csv',engine='python')
X=data.iloc[:,1].values


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
#data=data.iloc[:4631,:]
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])
data.to_excel('Posneg.xlsx')
value=[0,0,0]
#Checking for pos and neg

for i in range(463418):
    if(data['compound'][i]>0):
        value[0]+=1
    elif(data['compound'][i]==0):
        value[1]+=1
    else:
        value[2]+=1
        
#Pie Chart for Verified
labels ='Positive','Neutral','Negative'
sizes=value
fig1, ax1 = plt.subplots()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(sizes,labels=labels,labeldistance=1.2,autopct='%1.1f%%',pctdistance=0.7,wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
plt.rcParams['text.color'] = 'black'
p=plt.gcf()
#plt.pie(sizes, labels=names, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' })
p.gca().add_artist(my_circle)
ax.legend(labels=['Positive','Neutral','Negative'])  
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/posneg.png")
plt.show()
 






