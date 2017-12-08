import numpy as np
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Embedding , Dropout , LSTM , Dense , GRU , Flatten , Bidirectional
from keras.optimizers import Adam  
from keras.layers.convolutional import Conv1D , MaxPooling1D 
from keras.preprocessing.text import text_to_word_sequence 
from nltk.corpus import stopwords
from gensim import models 
import pickle
from keras.models import load_model
import sys
test_num = 200000
 
test_in = np.load(sys.argv[1])

model = load_model('model_w2v.h5')
print('predicting')  
result = model.predict(test_in) 
 
with open(sys.argv[2], 'w') as fp:
    fp.write('id,label\n')
    for i in range(test_num): 
        if(result[i]>0.5):
        	ans = 1
        else:
        	ans = 0	
        fp.write('%d,%d\n' % (i, ans)) 