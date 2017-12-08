import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer 
from gensim import models 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D, ZeroPadding2D , LSTM ,Embedding , Bidirectional , GRU  
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam  
from keras.models import load_model
from scipy import ndimage 
from keras.preprocessing.text import Tokenizer,text_to_word_sequence  
import sys
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn import feature_extraction  
import matplotlib.pyplot as plt 

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p] 
 
  
train_in = np.load(sys.argv[1]) 
labels = np.load(sys.argv[2]) 

unison_shuffled_copies(train_in,labels) 
vi = train_in[:30000]
vo = labels[:30000]
train_in = train_in[30000:]
labels = labels[30000:]  
 

model = Sequential()  
model.add(Bidirectional(GRU(256, recurrent_dropout=0.25, dropout=0.25, activation='tanh', return_sequences=True ), input_shape=(25,100))) 
model.add(Bidirectional(GRU(128, recurrent_dropout=0.25, dropout=0.25, activation='tanh', return_sequences=True ))) 
model.add(Bidirectional(GRU(64, recurrent_dropout=0.25, dropout=0.25, activation='tanh' ))) 
model.add(Dropout(0.5)) 
model.add(Dense(256 , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128 , activation = 'relu' ))
model.add(Dropout(0.4))
model.add(Dense(64 , activation = 'relu' ))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))  
adam = Adam(lr=0.001,decay=1e-5,clipvalue=0.5)  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
print(model.summary()) 
model.fit(train_in, labels, epochs=20, batch_size=512 , validation_data=(vi,vo)) 
print('saving model')
model.save('model_w2v_train.h5') 