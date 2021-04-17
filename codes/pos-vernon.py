import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('Posneg.csv',engine='python')
X=data.iloc[:,1].values

'''
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
'''
posvalue=[0,0]
negvalue=[0,0]
neuvalue=[0,0]
for i in range(463418):
    if(data['compound'][i]>0):
        if(data['verified'][i]==1):
            posvalue[0]+=1
        else:
            posvalue[1]+=1
    elif(data['compound'][i]<0):
        if(data['verified'][i]==1):
            negvalue[0]+=1
        else:
            negvalue[1]+=1
    else:
        if(data['verified'][i]==1):
            neuvalue[0]+=1
        else:
            neuvalue[1]+=1
#Donut for Positive
labels =['Verified','Non Verified']
sizes=posvalue
#fig1, ax1 = plt.subplots()
colors = ['#1515B4','#3BB9FF']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Positive')
p=plt.gcf()

plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/posvernon.png")
plt.show()

#Donut for negative

labels =['Verified','Non Verified']
sizes=negvalue
#fig1, ax1 = plt.subplots()
colors = ['#6F4E37','#C85A17']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Negative')
p=plt.gcf()

plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/negvernon.png")
plt.show()

#Donut for neutral

labels =['Verified','Non Verified']
sizes=neuvalue
#fig1, ax1 = plt.subplots()
colors = ['#EC83D9','#ED1FC4']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Neutral')
p=plt.gcf()

plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/neuvernon.png")
plt.show()

