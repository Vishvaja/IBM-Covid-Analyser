import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

data=pd.read_csv('Posneg.csv',engine='python')



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 463418):
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

'''
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])

'''



pos_review=[j for i,j in enumerate(data['text']) if data['compound'][i]>0.2]
neu_review=[j for i,j in enumerate(data['text']) if 0.2>=data['compound'][i]>=-0.2]
neg_review=[j for i,j in enumerate(data['text']) if data['compound'][i]<-0.2]
print("Percentage of positive review:{}%".format(len(pos_review)*100/len(data['text'])))
print("Percentage of neutral review:{}%".format(len(neu_review)*100/len(data['text'])))
print("Percentage of negative review:{}%".format(len(neg_review)*100/len(data['text'])))

X=data.iloc[:,1].values
length=[]
for i in range(463418):
    length.append(len(str(X[i])))

data['len']=length



Y=data.iloc[:,[7,5]].values
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

fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 10, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 10, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 30, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Length of tweet')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/clusterlength.png")

def isNaN(string):
    return string != string

for i in range(463418):
    if(isNaN(X[i])):
        X[i]="coronavirus"
data['text']=X

#Maximum features
a=data.iloc[:,1].values
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

l=[]
#data=data.iloc[:1000,:]
for i in range(463418):
    s=0
    for j in range(1500):
        s=s+X[i][j]
    l.append(s)

data['len']=l

Y=data.iloc[:,[7,5]].values
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

fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 1, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 1, c = 'blue', label = 'Cluster 2')
'''plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 1, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Max features')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/clustermax.png")

#Verified & Non Verified

X=data.iloc[:,0].values
vcount=0
ncount=0
for i in range(463418):
    if(X[i]==1):
        vcount+=1
    else:
        ncount+=1
        
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
colors=['#008080','#CACACA','#575757']
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

fig1.savefig('C:/Users/pjai1/OneDrive/Desktop/IBM/images/verifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#Pie Chart for Non-Verified    
labels = 'Positive','Neutral','Negative'
sizes=nlist
explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig2, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
fig2.savefig('C:/Users/pjai1/OneDrive/Desktop/IBM/images/nonverifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


