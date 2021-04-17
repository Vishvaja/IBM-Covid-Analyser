# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CORONA.CSV',engine="python")
Y=dataset.iloc[:,20].values
X=[]
Z=[]
langs = ['es','zh','en', 'fr','nl','el','ja','th','hi','ar','pt','tr','tl','fa','et','ta','de','it','in','ru','bn','gu','kn','or','te','sl','ne','ro','lt','mr','pl','ur','ml','ko','ca','pa','vi','da','no','si','sv','cs','fi','ht','iw','eu','bg','cy','hy','am','sr','is','hu','lv','ps','sd','dv','uk','km','lo','ckb','ka']

la=[]
for i in langs:
    c=0
    for j in range(463418):
        if(Y[j]==i):
            c=c+1
    X.append(c)
s=0
for i in X:
    s=s+i
    

summa={}
lan={}
for key in langs:
    for value in X:
        lan[key]=value
        X.remove(value)
        break
Z=[]
summa= sorted(lan.items(), key=lambda x: x[1], reverse=True)



for i in range(62):
    Z.append(summa[i][0])
    
for j in range(62):
    la.append(summa[j][1])

for i in X:
    s=s+i
Z=Z[:5]
la=la[:5]

fig,ax=plt.subplots()
ax.bar(Z[0],la[0],color="#000000")
ax.bar(Z[1],la[1],color="#0000CD")
ax.bar(Z[2],la[2],color="#00FFFF")
ax.bar(Z[3],la[3],color="#ADD8E6")
ax.bar(Z[4],la[4],color="#A9A9A9")     
ax.legend(labels=['English','Spanish','Italian','French','German']) 
plt.xlabel('Languages')
plt.ylabel('Tweets')      
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/languages.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

counts, bins = np.histogram(data)
plt.hist(bins[:-1], bins, weights=counts)

import word2vec

w2v_model = word2vec(min_count=3,
                     window=4,
                     size=300,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)

word_vectors = word2vec.load("word2vec.model").wv
model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
positive_cluster_center = model.cluster_centers_[0]
negative_cluster_center = model.cluster_centers_[1]
words = pd.DataFrame(word_vectors.vocab.keys())
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: word_vectors.wv[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])
words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value




#Converting language to english
from googletrans import Translator
translator = Translator()
dataset["City_English"] = dataset["City_trad_chinese"].map(lambda x: translator.translate(x, src="zh-TW", dest="en").text)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 10):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10)
Z = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:, 1].values
V= [[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]

for i in range(10):
    for j in range(10):
        V[i][j]=Z[j][i]
# Using the elbow method to find the optimal number of clusters
    
arr=np.array(V)
np.concatenate((Z,Y),axis=1)
for i in range(10):
    for j in range(11):
        Z[i][10]=Y[i]
from sklearn.cluster import KMeans
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


kmeans = KMeans(n_clusters =3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)


plt.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
'''plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')'''
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('People')
plt.ylabel('Reaction to Corona')
plt.legend()
plt.show()


'''# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(X, y_kmeans)




import word2vec
import gensim

model=gensim.models.Word2Vec(
        a,
        size=150,
        window=10,
        min_count=2,
        workers=10,iter=10)

model.build_vocab(corpus)
a=['globe','sphere','world']
from gensim.models.word2vec import LineSentence
models123=gensim.models.Word2Vec(LineSentence(corpus), size=100, window=5, min_count=2, workers=4)


w1="world"
model.wv.most_similar(positive=w1)

