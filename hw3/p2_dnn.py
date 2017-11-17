import numpy as np
import sys
import csv 

from scipy import ndimage

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam  
from keras.models import load_model
from scipy import ndimage
import matplotlib.pyplot as plt 

sample_num = 28709
test_num = 7178
train_in=[]
train_out=[] 

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

with open(sys.argv[1], 'r', encoding='UTF-8') as in_file:
	in_file.readline()
	for i in range(sample_num): 
		f = in_file.readline().replace('\n','').split(',') 
		target = [0 for z in range(7)]
		target[int(f[0])] = 1;
		split = f[1].split(" ") 
		feature = [float(x)/255 for x in split]  
		train_in.append(feature)
		train_out.append(target)  
     

train_in = np.array(train_in) 
train_out = np.array(train_out)  

train_in = train_in.reshape(train_in.shape[0],48*48)  

############################################################################
model = Sequential()  
model.add(Dense(input_dim=48*48,units=128,activation='relu'))
model.add(Dropout(0.5))    
model.add(Dense(units=7,activation='relu'))
model.add(Activation('softmax')) 
model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

train_history = model.fit(x=train_in, y=train_out, validation_split=0.2, epochs=100, batch_size=200, verbose=2)  
#show_train_history(train_history, 'acc', 'val_acc') 

test_in=[] 
with open(sys.argv[2], 'r', encoding='UTF-8') as in_file:
	in_file.readline()
	for i in range(test_num): 
		f = in_file.readline().replace('\n','').split(',')  
		split = f[1].split(" ")  
		feature = [float(x)/255 for x in split] 
		test_in.append(feature) 
test_in = np.array(test_in)  
test_in = test_in.reshape(test_in.shape[0],48,48,1) 
result = model.predict(test_in) 
with open(sys.argv[3], 'w') as fp:
    fp.write('id,label\n')
    for i in range(test_num): 
        max_ = 0
        ans = -10
        for j in range(7):
            if result[i][j] > max_:
                max_ = result[i][j]
                ans = j
        fp.write('%d,%d\n' % (i, ans))



		 
