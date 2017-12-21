import numpy as np 
import keras 
import os
from keras.models import load_model
import sys 

avg = 3.58171208604
std = 1.11689766115

user_id = []
movie_id = []
with open(sys.argv[1], 'r', encoding='UTF-8') as fp:
	fp.readline()
	for i in range (100336):
		a = fp.readline().replace('\n','').split(',')
		user_id.append(int(a[1]))
		movie_id.append(a[2])

user_id = np.array(user_id)
movie_id = np.array(movie_id)

model1 = load_model('model_mf_64.h5')
model2 = load_model('model_mf_128.h5')
model3 = load_model('model_mf_256.h5')
model4 = load_model('model_mf_512.h5') 

result1 = model1.predict([user_id,movie_id])
result2 = model2.predict([user_id,movie_id])
result3 = model3.predict([user_id,movie_id])
result4 = model4.predict([user_id,movie_id]) 

result1 = ( result1 * std ) + avg
result2 = ( result2 * std ) + avg
result3 = ( result3 * std ) + avg
result4 = ( result4 * std ) + avg 



def cut(x):
	if( x > 5 ):
		return 5
	elif( x < 1 ):
		return 1
	else:
		return x

with open(sys.argv[2], 'w') as fp:
	fp.write('TestDataID,Rating\n')
	for i in range(100336): 
		ans = float ( cut(result1[i]) + cut(result2[i]) + cut(result3[i]) + cut(result4[i]) ) / 4
		fp.write('%d,%f\n' % (i+1, ans))
