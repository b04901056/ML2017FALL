import sys
import numpy as np
import pandas as pd
import csv
d = np.genfromtxt(sys.argv[1], delimiter=',')
data = d[1:,3:]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0 
all_data=np.empty([ 18, 24 * 20 * 12], dtype=float)

for i in range(1, 20 * 12 + 1 ):
    all_data[:, 24*(i-1) : 24*i] = data[ 18 *(i-1): 18*i , :]

def get_month_data(i):
    return all_data[:, 480 *(i-1): 480 * i]

def normalize(b):
    array=np.array(b,dtype=float)
    row_means = np.mean(array, axis=0)
    row_std = np.std(array, axis=1)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if not row_std[i]== 0 :
                array[i][j] = (array[i][j]- row_means[j]) / row_std[j]
    #return array
    return(row_means,row_std,array)
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
  
##MODEL##  

def loss(theta , x , y ):
     return np.sum(np.power(x.dot(theta) - y, 2 ))/ 5652

def gradient_descent( x , y  , repeat ):
    dim=x.shape[1]+1 
    theta=np.zeros((dim, 1 ))
    one_x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1)
    learning_rate= np.array([[200]] * dim)
    tot=np.zeros((dim, 1 ))
    lamda=0.1
    for T in range(repeat):
        #if(T% 2000 == 0 ):
            #print("T=",T)
            #print("Loss:",loss(theta,one_x,y))
        tot+=(np.transpose(one_x).dot(y-one_x.dot(theta)))**2
        #theta = theta - learning_rate *( (-2) * (np.transpose(one_x).dot(y-one_x.dot(theta)))+ 2 *lamda*theta)/(np.sqrt(tot) + 0.005)
        theta = theta - learning_rate *( (-2) * (np.transpose(one_x).dot(y-one_x.dot(theta))))/(np.sqrt(tot) + 0.005)
    return theta
 
def feature(x,row,number,power,test):
    count=0
    par=np.empty([ 5652 , number ],dtype=float)
    for i in range( 5751 ):
        if ((i% 480 )<= 470 ):
            par[count, : ]= all_data[ row , i+9-number:i+9 ]
            count+=1
    x=np.concatenate( (x , np.power( par , power ))  , axis = 1)
    test=np.concatenate( (test,[[row,number,power]] )  , axis = 0)
    return (x,test)

#################Test###############

y=np.empty([ 5652 , 1 ],dtype=float)
count=0
for i in range(5751):
    if ((i% 480 )<= 470 ):
        y[count, 0 ]=all_data[ 9 , i+9 ] 
        count+=1
 
d = np.genfromtxt(sys.argv[2], delimiter=',')
data=d[:, 2: ]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0
title = [['id','value']]
f = open(sys.argv[3],"w")
w = csv.writer(f)
w.writerows(title)
predict=np.empty([ 240 , 1 ],dtype=float)

x=np.empty([ 5652 , 1 ],dtype=float)
test=np.empty([ 1 , 3 ],dtype=int)

#feature(x,第幾個row,取的數量,次方,test):
a=feature(x,9,9,1,test)
x=a[0]
test=a[1]

a=feature(x,9,4,2,test)
x=a[0]
test=a[1]

a=feature(x,8,9,1,test)
x=a[0]
test=a[1]

a=feature(x,8,4,2,test)
x=a[0]
test=a[1]

a=feature(x,7,3,1,test)
x=a[0]
test=a[1]

#a=feature(x,12,2,1,test)
#x=a[0]
#test=a[1]

a=feature(x,16,2,1,test)
x=a[0]
test=a[1]

##########################################################

a=feature(x,10,1,1,test)
x=a[0]
test=a[1]


a=feature(x,11,2,1,test)
x=a[0]
test=a[1]

a=feature(x,4,2,1,test)
x=a[0]
test=a[1]

a=feature(x,0,2,1,test)
x=a[0]
test=a[1]

 
###########################################
x=x[:,1:]
test=test[1:,:]
rec_max=get_max(x)
rec=normalize(x)
#input_x=divide_max(x)
input_x=normalize(x)[2]
print(input_x)
result=gradient_descent( input_x , y  , 100000 )

for i in range(int(data.shape[0]/18)):
    test_input=[1]
    for j in range(test.shape[0]):
        test_input=np.append(test_input,np.power(data[ (test[j][0])+18*i , 9-test[j][1]: ], test[j][2] ))
    test_input=np.array(test_input,dtype=float).reshape(1,x.shape[1]+1)
    for j in range(1,test_input.shape[1]):
        #test_input[0][j]=(test_input[0][j])/float(rec_max[ j-1 ])
        test_input[0][j]=(test_input[0][j]-rec[0][j-1])/rec[1][j-1]
    predict[i][0]=float(test_input.dot(result))

for i in range(predict.shape[0]):
    title=[["id_"+str(i),float(predict[i][0])]]
    w.writerows(title)

f.close()
 
