import numpy as np

dataset=pd.read_csv('Posneg.csv',nrows=100000,engine="python")
X=dataset.iloc[:,5].values
Y=dataset.iloc[:,6].values


p=[0,0,0,0]
nu=[0,0,0,0]
ne=[0,0,0,0]
for i in range(100000):
    if(X[i]>0):
        if(Y[i]=='sad'):
            p[0]=p[0]+1
        elif(Y[i]=='happy'):
            p[1]=p[1]+1
        elif(Y[i]=='angry'):
            p[2]=p[2]+1
        else:
            p[3]=p[3]+1
    elif(X[i]==0):
        if(Y[i]=='sad'):
            nu[0]=nu[0]+1     
        elif(Y[i]=='happy'):
            nu[1]=nu[1]+1
        elif(Y[i]=='angry'):
            nu[2]=nu[2]+1
        else:
            nu[3]=nu[3]+1
    elif(X[i]<0):
        if(Y[i]=='sad'):
            ne[0]=ne[0]+1
        elif(Y[i]=='happy'):
            ne[1]=ne[1]+1
        elif(Y[i]=='angry'):
            ne[2]=ne[2]+1
        else:
            ne[3]=ne[3]+1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x3 = [1,2,3,4,5,6,7,8,9]
y3 = [5,6,7,8,2,5,6,3,7]
z3 = np.zeros(9)

dx = np.ones(9)
dy = np.ones(9)
dz = [1,2,3,4,5,6,7,8,9]

ax1.bar3d(x3, y3, z3, dx, dy, dz)


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
'''
np.arange(nu)
np.arange(p)
np.arange(ne)
x3 = nu
y3 =p
z3 = np.zeros(4)
'''
x3=np.asarray(nu)
y3 =np.asarray(p)
z3 = np.zeros(4)

dx = np.ones(4)
dy = np.ones(4)
dz = np.asarray(ne)
'''
x3=np.arange()
y3=np.arange()
dz=np.arange()
'''
ax1.bar3d(x3, y3, z3, dx, dy, dz)


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()