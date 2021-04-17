import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv('Posneg.csv',nrows=100000,engine='python')


Y=dataset.iloc[:100000,6].values
X=dataset.iloc[:100000,0].values
labels=['sad','happy','angry','informative']

v=[0,0,0,0]
n=[0,0,0,0]

for i in range(100000):
    if(X[i]==1):
        if(Y[i]=='sad'):
            v[0]=v[0]+1
        elif(Y[i]=='happy'):
            v[1]=v[1]+1
        elif(Y[i]=='angry'):
            v[2]=v[2]+1
        else:
            v[3]=v[3]+1
    else:
        if(Y[i]=='sad'):
            n[0]=n[0]+1
        elif(Y[i]=='happy'):
            n[1]=n[1]+1
        elif(Y[i]=='angry'):
            n[2]=n[2]+1
        else:
            n[3]=n[3]+1
            
#For verified
fig,ax=plt.subplots()
ax.bar(labels[0],v[0],color="#000000")
ax.bar(labels[1],v[1],color="#0000CD")
ax.bar(labels[2],v[2],color="#00FFFF")
ax.bar(labels[3],v[3],color="#ADD8E6")     
ax.legend(labels=['sad','happy','angry','informative']) 
plt.xlabel('Emotions')
plt.ylabel('No. of accounts')   
plt.title('Emotions of Verified Users')   
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/labelsvsverified.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#For non verified
fig,ax=plt.subplots()
ax.bar(labels[0],n[0],color="#000000")
ax.bar(labels[1],n[1],color="#0000CD")
ax.bar(labels[2],n[2],color="#00FFFF")
ax.bar(labels[3],n[3],color="#ADD8E6")     
ax.legend(labels=['sad','happy','angry','informative']) 
plt.xlabel('Emotions')
plt.ylabel('No. of accounts')   
plt.title('Emotions of Non-Verified Users')    
plt.show()
fig.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/labelsvsnv.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

        

