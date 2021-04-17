import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Posneg.csv",engine="python",nrows=100000)
X=dataset.iloc[:,5]
Y=dataset.iloc[:,7]
for i in range(100000):
    if(X[i]<0):
        Y[i]=-1
    elif(X[i]==0):
        Y[i]=0
    else:
        Y[i]=1
        
dataset['len']=Y


Z=dataset.iloc[:100,[5,6]].values

W=dataset.iloc[:,6].values
        
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
W = labelencoder_X.fit_transform(W)
onehotencoder = OneHotEncoder(categories=[0])
W = onehotencoder.fit_transform(W).toarray()
dataset['new']=W
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(Z)
fig,ax=plt.subplots()
# Visualising the clusters
ax.scatter(Z[y_kmeans == 0, 0], Z[y_kmeans == 0, 1],marker='*',s = 100, c = 'red', label = 'Angry')
ax.scatter(Z[y_kmeans == 1, 0], Z[y_kmeans == 1, 1],marker='P', s = 30, c = 'pink', label = 'Sad')
ax.scatter(Z[y_kmeans == 2, 0], Z[y_kmeans == 2, 1],marker='8', s = 30, c = '#d8bfd8', label = 'Informative')
ax.scatter(Z[y_kmeans == 3, 0], Z[y_kmeans == 3, 1],marker='2', s = 100, c = 'grey',label = 'Happy' )
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Sentiments VS Emotions')
plt.xlabel('Emotion')
plt.ylabel('Sentiment score')
ax.legend(loc=0,framealpha=0.1)
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/sentimentscorelabels.png",dpi=100, bbox_inches='tight', pad_inches=0.0)