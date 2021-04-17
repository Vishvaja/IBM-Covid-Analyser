import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

data=pd.read_csv('Translated.csv',engine='python')



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


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
#data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])'
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])





pos_review=[j for i,j in enumerate(data['text']) if data['compound'][i]>0.2]
neu_review=[j for i,j in enumerate(data['text']) if 0.2>=data['compound'][i]>=-0.2]
neg_review=[j for i,j in enumerate(data['text']) if data['compound'][i]<-0.2]
print("Percentage of positive review:{}%".format(len(pos_review)*100/len(data['text'])))
print("Percentage of neutral review:{}%".format(len(neu_review)*100/len(data['text'])))
print("Percentage of negative review:{}%".format(len(neg_review)*100/len(data['text'])))

a=data.iloc[:1000,1].values

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=250)
X=cv.fit_transform(a).toarray()

l=[]
data=data.iloc[:1000,:]
for i in range(1000):
    s=0
    for j in range(250):
        s=s+X[i][j]
    l.append(s)

data['len']=l

Y=data.iloc[:,[6,5]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)


plt.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 1, c = 'red', label = 'Cluster 1')
plt.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 1, c = 'blue', label = 'Cluster 2')
'''plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 1, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')'''
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 10, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Length of tweet')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()



