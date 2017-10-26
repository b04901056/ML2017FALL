import numpy as np
import sys
import csv 
x = np.genfromtxt(sys.argv[1], delimiter=',')
data = x[1:,:]
s = np.genfromtxt(sys.argv[2], delimiter=',')
y = np.empty([ len(s)-1 , 1 ],dtype=float)
for i in range(1,len(s)):
    y[i-1, 0 ]=s[i]

def get_max(ipt):
    i=np.array(ipt,dtype=float)
    return i.max(axis=0)
    
def divide_max(ipt):
    ipt=np.array(ipt,dtype=float)
    rec_max=ipt.max(axis=0)
    for i in range(ipt.shape[0]):
        for j in range(ipt.shape[1]):
            ipt[i][j]=float(ipt[i][j]/rec_max[j])
    return ipt

def func(x,theta):
    z=np.empty([x.shape[0],1],dtype=float)
    for i in range(x.shape[0]):
        t=x[i,:].dot(theta)
        t=t*(-1)
        z[i][0]=1/(1+np.exp(t))
    return z

def loss(x,y,theta):
    pre=func(x,theta)
    pre=np.array(pre,dtype=float)
    for i in range(pre.shape[0]):
        if pre[i][0]<1e-323:
            pre[i][0]=1e-323       
    a=np.transpose(y).dot(np.log(pre))
    one=np.ones([x.shape[0],1],dtype=float)
    pre=one-pre
    for i in range(pre.shape[0]):
        if pre[i][0]<1e-323:
            pre[i][0]=1e-323
    b=(np.transpose(one)-np.transpose(y)).dot(np.log(pre))
    return (-1)*(a+b) 

#feature( x , 第幾個參數 , 次方 , test )

def feature( x , num , power , test ):
    par=np.empty([ x.shape[0] , 1 ],dtype=float)
    par[:,0]=data[:,num]
    x=np.concatenate((x , np.power( par , power )), axis =  1 )
    test=np.concatenate( (test , [[ num , power ]]), axis =  0 )
    return (x,test)
 
def answer(x):
    ans=np.empty([x.shape[0],1],dtype=int)
    for i in range(x.shape[0]):
        if x[i]>0.5:
            ans[i]=1;
        else:
            ans[i]=0;
    return ans

def accu(ans,y):
    num=ans.shape[0]
    count=0
    for i in range(num):
        if ans[i]==y[i]:
            count+=1
    return count/num

def gradient_descent( one_x , y , repeat ):
    dim=one_x.shape[1]
    theta=np.zeros((dim, 1 ))
    learning_rate= np.array([[1]] * dim)
    tot=np.zeros((dim, 1 ))
    lamda=0.1 
    for T in range(repeat):
        if(T% 1 == 0 ):
            #print("T=",T)
            #print("acc: ",accu(answer(func(one_x,theta)),y),"loss:",float(loss(one_x,y,theta)))
        add=( np.transpose(one_x).dot(y-func(one_x,theta)))
        tot+=add*add
        theta -= learning_rate *( (-1) * add + 2 *lamda*theta)/(tot**0.5)
    return theta

test=np.empty([ 1 , 2 ],dtype=float)
x_use=np.empty([ data.shape[0] , 1 ],dtype=float)
##############################################調參數#######################################
 
for i in range(106):
    a=feature(x_use,i,1,test)
    x_use=a[0]
    test=a[1]

for i in range(0,6):
    if i == 1 or i == 2:
        continue
    for j in range(1,20):    
        if j==2:
            continue
        a=feature(x_use,i,j/2,test)
        x_use=a[0]
        test=a[1]
for i in range(6):
    if i == 2:
        continue
    par=np.empty([ x_use.shape[0] , 1 ],dtype=float)
    par[:,0]=data[:,i]
    x_use=np.concatenate((x_use , np.log(1+par) ), axis =  1 )

############################################################################################
x_use=x_use[:,1:]
test=test[1:,:]
rec_max=get_max(x_use)
x_use=divide_max(x_use)
x_use=np.concatenate((np.ones((x_use.shape[0], 1 )), x_use) , axis = 1)
a=gradient_descent(x_use,y,1600)
b=func(x_use,a)
ans=answer(b)
#print("training accuracy: ",accu(ans,y))
############################################################################### 
s = np.genfromtxt(sys.argv[3], delimiter=',')
s = s[1:,:]
x_use=np.empty([s.shape[0],test.shape[0]+5],dtype=float)
for i in range(test.shape[0]):
    x_use[:,i]=np.power(s[:,int(test[i][0])],test[i][1])

x_use[:,test.shape[0]]=np.log(1+s[:,0])
x_use[:,test.shape[0]+1]=np.log(1+s[:,1])
x_use[:,test.shape[0]+2]=np.log(1+s[:,3])
 
x_use[:,test.shape[0]+3]=np.log(1+s[:,4])

x_use[:,test.shape[0]+4]=np.log(1+s[:,5])

for i in range(x_use.shape[0]):    
    for j in range(x_use.shape[1]):
        x_use[i][j]=x_use[i][j]/rec_max[j]

x_use=np.concatenate((np.ones((x_use.shape[0], 1 )), x_use) , axis = 1)
b=func(x_use,a)
ans=answer(b)

f = open(sys.argv[4],"w")
w = csv.writer(f)
title = [['id','label']]
w.writerows(title)

for i in range(ans.shape[0]):
    title=[[str(i+1),int(ans[i][0])]]
    w.writerows(title)

f.close()



 



















