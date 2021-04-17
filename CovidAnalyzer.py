#COVID ANALYZER
data.to_csv('output.csv')
#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

#Importing Dataset

data=pd.read_csv('CORONA.CSV',engine='python')
X=data.iloc[:,4].values

#Filling missing data

def isNaN(string):
    return string != string

for i in range(463418):
    if(isNaN(X[i])):
        X[i]="coronavirus"
data['text']=X

#Cleaning the texts

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


#Obtaining sentiment scores
    
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['neg']=data['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
data['neu']=data['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
data['pos']=data['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
data['compound']=data['text'].apply(lambda x:sia.polarity_scores(x)['compound'])

#Percentage values of sentiments

pos_review=[j for i,j in enumerate(data['text']) if data['compound'][i]>0.2]
neu_review=[j for i,j in enumerate(data['text']) if 0.2>=data['compound'][i]>=-0.2]
neg_review=[j for i,j in enumerate(data['text']) if data['compound'][i]<-0.2]
print("Percentage of positive review:{}%".format(len(pos_review)*100/len(data['text'])))
print("Percentage of neutral review:{}%".format(len(neu_review)*100/len(data['text'])))
print("Percentage of negative review:{}%".format(len(neg_review)*100/len(data['text'])))

#CLUSTERING USING LENGTH

#Finding length of tweets

length=[]
for i in range(463418):
    length.append(len(str(X[i])))
data['len']=length

#Elbow method 

'''
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
'''

#K-Means for Clustering using length

Y=data.iloc[:,[17,16]].values
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
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Length of tweet')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/clusterlength.png")

#CLUSTERING USING MAX FEATURES

#Obtaining bag of words model

cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

#Finding number of max features

l=[]
for i in range(463418):
    s=0
    for j in range(1500):
        s=s+X[i][j]
    l.append(s)
data['len']=l

#Elbow method

'''
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
'''

#K-Means for Clustering using max features

kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)
fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 1, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 1, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 1, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Max features')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/clustermax.png")

#VERIFIED AND NON-VERIFIED USING SENTIMENT SCORES

#Obtaining verified and non-verified accounts count

X=data.iloc[:,11].values
vcount=0
ncount=0
for i in range(463418):
    if(X[i]==1):
        vcount+=1
    else:
        ncount+=1
        
#Count of positive,negative and neutral tweets 
        
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
 
#Pie Chart for Verified Tweets
        
labels = 'Positive','Neutral','Negative'
sizes=vlist
explode = (0.1, 0, 0)  
fig1, ax1 = plt.subplots()
colors=['#008080','#CACACA','#575757']
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
fig1.savefig('C:/Users/pjai1/OneDrive/Desktop/IBM/images/verifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#Pie Chart for Non-Verified Tweets
    
labels = 'Positive','Neutral','Negative'
sizes=nlist
explode = (0.1, 0, 0) 
fig2, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()
fig2.savefig('C:/Users/pjai1/OneDrive/Desktop/IBM/images/nonverifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#LANGUAGE DOMINANCE

#Finding number of tweets in each language

Y=dataset.iloc[:,12].values
X=[]
Z=[]
langs=['es','zh','en', 'fr','nl','el','ja','th','hi','ar','pt','tr','tl','fa','et','ta','de','it','in','ru','bn','gu','kn','or','te','sl','ne','ro','lt','mr','pl','ur','ml','ko','ca','pa','vi','da','no','si','sv','cs','fi','ht','iw','eu','bg','cy','hy','am','sr','is','hu','lv','ps','sd','dv','uk','km','lo','ckb','ka']
la=[]
for i in langs:
    c=0
    for j in range(463418):
        if(Y[j]==i):
            c=c+1
    X.append(c)

#Creating a dictionary   

dic={}
lan={}
for key in langs:
    for value in X:
        lan[key]=value
        X.remove(value)
        break
Z=[]
dic= sorted(lan.items(), key=lambda x: x[1], reverse=True)
for i in range(62):
    Z.append(dic[i][0])
for j in range(62):
    la.append(dic[j][1])
    
#Finding top five languages count  
    
Z=Z[:5]
la=la[:5]
fig,ax=plt.subplots()
ax.bar(Z[0],la[0],color="#000000")
ax.bar(Z[1],la[1],color="#0000CD")
ax.bar(Z[2],la[2],color="#00FFFF")
ax.bar(Z[3],la[3],color="#ADD8E6")
ax.bar(Z[4],la[4],color="#A9A9A9")     
ax.legend(labels=la) 
plt.xlabel('Languages')
plt.ylabel('Tweets')      
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/languages.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#VERIFIED AND NON-VERIFIED

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
            
#Donut for Positive Sentiments
            
labels =['Verified','Non Verified']
sizes=posvalue
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

#Donut for Negative Sentiments

labels =['Verified','Non Verified']
sizes=negvalue
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

#Donut for Neutral Sentiments

labels =['Verified','Non Verified']
sizes=neuvalue
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

#TOTAL SENTIMENTS

value=[0,0,0]
for i in range(463418):
    if(data['compound'][i]>0):
        value[0]+=1
    elif(data['compound'][i]==0):
        value[1]+=1
    else:
        value[2]+=1
        
#Plotting overall sentiments
        
labels ='Positive','Neutral','Negative'
sizes=value
fig1, ax1 = plt.subplots()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(sizes,labels=labels,labeldistance=1.2,autopct='%1.1f%%',pctdistance=0.7,wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
plt.rcParams['text.color'] = 'black'
p=plt.gcf()
p.gca().add_artist(my_circle)
ax.legend(labels=['Positive','Neutral','Negative'])  
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/posneg.png")
plt.show()

#IMPACT CLUSTERING

#Elbow Method
'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''
#K Means Clustering for followers vs favourite count

Y=dataset.iloc[:,[8,6]].values
kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)
fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 30, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.ylabel('No. of Followers')
plt.xlabel('Favourites count')
plt.legend()
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/impactcluster.png",dpi=100, bbox_inches='tight', pad_inches=0.0)


#C


 