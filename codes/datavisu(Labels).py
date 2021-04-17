
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Posneg.csv',nrows=100,engine='python')

X=dataset.iloc[:,6].values
Y=dataset.iloc[:,8].values

fig,ax=plt.subplots()
#plt.scatter(X_train, y_train, color = 'red')
ax.scatter(Y,X,color='red')
ax.plot(Y,X)
plt.xlabel("Labels")
plt.ylabel("Verified or non verified")
plt.show()

fig=savefig("")
