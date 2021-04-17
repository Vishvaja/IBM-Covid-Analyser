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

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
#legend([plot1], "title", prop=fontP)

import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
y1 = [2, 3, 4.5]
y2 = [1, 1.5, 5]

plt.plot(y1)
plt.plot(y2)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["blue", "green"], loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.savefig("out.png")

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(Z)

# Visualising the clusters
plt.scatter(Z[y_kmeans == 0, 0], Z[y_kmeans == 0, 1],marker='*',s = 100, c = 'red', label = 'Angry')
plt.scatter(Z[y_kmeans == 1, 0], Z[y_kmeans == 1, 1],marker='P', s = 70, c = 'blue', label = 'Sad')
plt.scatter(Z[y_kmeans == 2, 0], Z[y_kmeans == 2, 1],marker='8', s = 30, c = 'green', label = 'Informative')
plt.scatter(Z[y_kmeans == 3, 0], Z[y_kmeans == 3, 1],marker='2', s = 500, c = 'black',label = 'Happy' )
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Sentiments VS Emotions')
plt.xlabel('Emotion')
plt.ylabel('Sentiment score')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1.5
legend_y = 0.5
plt.legend(loc="center right",bbox_to_anchor=(legend_x, legend_y))
plt.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/sentivsemo.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
