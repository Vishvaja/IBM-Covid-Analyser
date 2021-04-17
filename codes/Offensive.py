import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv('Posneg.csv',nrows=100000,engine='python')


Y=dataset.iloc[:100000,6].values
X=dataset.iloc[:100000,0].values
labels=['sad','happy','angry','informative']

Z=[]

for i in labels:
    c=0
    for j in range(100000):
        if(Y[j]==i):
            c=c+1
    Z.append(c)
 svno=[0,0]   
for i in range(100000):
    if(Y[i]=='sad'):
        if(X[i]==1):
           svno[0]=svno[0]+1
        else:
            svno[1]=svno[1]+1
            
#Pie Chart for Sad
labels = 'Verified','Non-Verified'
sizes=svno
explode = (0.1, 0)  
colors=['#A74AC7','#461B7E']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=180)
ax1.axis('equal') 
plt.title('Sad comments') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig1.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Sad.png")

#For happy

 hvno=[0,0]   
for i in range(100000):
    if(Y[i]=='happy'):
        if(X[i]==1):
            hvno[0]=hvno[0]+1
        else:
            hvno[1]=hvno[1]+1
            
#Pie Chart for Happy
labels = 'Verified','Non-Verified'
sizes=hvno
explode = (0.1, 0)  
colors=['#FFDB58','#8B4513']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=180)
ax1.axis('equal') 
plt.title('Happy comments') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig1.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Happy.png")

# For Angry
 avno=[0,0]   
for i in range(100000):
    if(Y[i]=='angry'):
        if(X[i]==1):
            avno[0]=avno[0]+1
        else:
            avno[1]=avno[1]+1
            
#Pie Chart for Angry
labels = 'Verified','Non-Verified'
sizes=avno
colors=['#F75D59','#CC0000']
explode = (0.1, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=180)
ax1.axis('equal') 
plt.title('Angry comments') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig1.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Angry.png")

#For informative

 ivno=[0,0]   
for i in range(100000):
    if(Y[i]=='informative'):
        if(X[i]==1):
            ivno[0]=avno[0]+1
        else:
            ivno[1]=avno[1]+1
            
#Pie Chart for Informtive
labels = 'Verified','Non-Verified'
sizes=ivno
colors=['#FFB6C1','#FF1493']
explode = (0.1, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=180)
ax1.axis('equal') 
plt.title('Informative comments') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig1.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Informative.png")

#Pie Chart for Labels
labels = 'Angry','Happy','Sad','Informative'
sizes=Z
explode = (0, 0.1,0,0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True,startangle=180)
ax1.axis('equal') 
plt.title('Tweet emotions') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig1.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Labels.png")


#Donut Chart


labels =['Angry','Happy','Sad','Informative']
sizes=Z
#fig1, ax1 = plt.subplots()
colors = ['#EC83D9','#A74AC7','#461B7E','#ED1FC4']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc=1)
plt.title('Tweet Emotions')
p=plt.gcf()

plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':2, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("C:/Users/pjai1/OneDrive/Desktop/IBM/images/Labels1.png")
plt.show()
