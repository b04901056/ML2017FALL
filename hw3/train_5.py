import numpy as np
import sys
import csv 


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam  
from keras.models import load_model

sample_num = 28709 
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
   
for i in range(sample_num) :
	  train_out.append(train_out[i])  

for i in range(sample_num):  
	reverse = []
	for j in range(1,49):
		for k in range(1,49):
			reverse.append(train_in[i][48*j-k]) 
	train_in.append(reverse)  

train_in = np.array(train_in) 
train_out = np.array(train_out)  

train_in = train_in.reshape(train_in.shape[0],48,48,1)  

############################################################################
model = Sequential()

model.add(Conv2D(32,3,3, activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(32,3,3, activation='relu'))
model.add(AveragePooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,3,3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(AveragePooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128,3,3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128,3,3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
 
model.add(Flatten()) 
 
model.add(Dropout(0.5))    
model.add(Dense(7))
model.add(Activation('softmax')) 
model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

train_history = model.fit(x=train_in, y=train_out, validation_split=0.2, epochs=150, batch_size=200, verbose=2)  
show_train_history(train_history, 'acc', 'val_acc') 






		 
