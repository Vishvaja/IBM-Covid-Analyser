
#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset

dataset=pd.read_csv('Classes.csv',engine='python',nrows=1000)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    sw=stopwords.words('english')
    sw.remove('not')
    review = [ps.stem(word) for word in review if not word in set(sw) ]
    review = ' '.join(review)
    corpus.append(review)

#dataset['xyz']=corpus
corpus=dataset.iloc[:999,0].values
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:999,1].values
'''
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#y = le.fit_transform(y)
y = le.fit_transform(y.astype(str))
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

i=0
X_train=[]
y_train=[]
y_test=[]
X_test=[]
for i in range (950,1000):
    y_test.append(y[i])
for i in range (950):
    y_train.append(y[i])
for i in range (950,1000):
    X_test.append(X[i])
for i in range (950):
    X_train.append(X[i])

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))









