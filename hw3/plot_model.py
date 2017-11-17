import numpy as np
import sys
import csv 
import pickle

from scipy import ndimage

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam  
from keras.models import load_model
from scipy import ndimage
import matplotlib.pyplot as plt 
 

def show_train_history(train_history, train, validation):  
    plt.plot(train_history[train])  
    plt.plot(train_history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

loaded = np.load('history_furfur.npy') 
train_history = loaded[()]
show_train_history(train_history, 'acc', 'val_acc') 





		 
