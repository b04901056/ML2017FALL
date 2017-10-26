import numpy as np
from numpy.linalg import inv
import sys
import csv 
x = np.genfromtxt(sys.argv[1], delimiter=',')
data = x[1:,:]
y = np.genfromtxt(sys.argv[2], delimiter=',')
y=y[1:]


def accu(ans,y):
    num=len(ans)
    count=0
    for i in range(num):
        if ans[i]==y[i]:
            count+=1
    return count/num

def normalize(b):
    array=np.array(b,dtype=float)
    row_means = np.mean(array, axis=0)
    row_std = np.std(array, axis=1)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if not row_std[i]== 0 :
               array[i][j] = (array[i][j]- row_means[j]) / row_std[j]
    return array


def divide_max(ipt):    
    ipt=np.array(ipt,dtype=float)
    rec_max=ipt.max(axis=0)
    for i in range(ipt.shape[0]):
        for j in range(ipt.shape[1]):
            if not rec_max[j]==0:
                ipt[i][j]=ipt[i][j]/rec_max[j]
    return ipt

arr0 = []
arr1 = []

for i in range(len(y)):
    if y[i]==0:
        arr0.append(data[i,:])
    else:
        arr1.append(data[i,:])

arr0=np.array(arr0)
arr1=np.array(arr1)
arr0=normalize(arr0)
arr1=normalize(arr1)


mean_0=np.mean(arr0,axis=0)
mean_1=np.mean(arr1,axis=0)
cov_0=np.cov(arr0.T)
cov_1=np.cov(arr1.T)

cov=(cov_0*arr0.shape[0]+cov_1*arr1.shape[0])/(arr0.shape[0]+arr1.shape[0])


w=np.transpose(mean_0-mean_1) @ inv(cov)
w=np.transpose(w)

b=(-0.5)*(mean_0.T) @ inv(cov) @ mean_0 + 0.5 * (mean_1.T) @ inv(cov) @mean_1 + np.log(arr0.shape[0]/arr1.shape[0])

def func(x,theta,b):
    z=np.empty([x.shape[0],1],dtype=float)
    for i in range(x.shape[0]):
        t=x[i,:] @ theta + b
        t=t*(-1)
        z[i][0]=1/(1+np.exp(t))
    return z

test = np.genfromtxt(sys.argv[3], delimiter=',')
test = test[1:,:]
arr = func(test,w,b)

result=[]
count=0
for i in range(arr.shape[0]):
    if arr[i]>0.5:
        result.append(0)
        count+=1
    else:
        result.append(1)

f = open(sys.argv[4],"w")
w = csv.writer(f)
title = [['id','label']]
w.writerows(title) 
for i in range(len(result)):
    title=[[str(i+1),int(result[i])]]
    w.writerows(title)

f.close()

 










