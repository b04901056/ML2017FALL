import numpy as np
import sys
import csv 
from keras.models import Sequential 
from keras.models import load_model

test_num = 7178
test_in=[] 

with open(sys.argv[1], 'r', encoding='UTF-8') as in_file:
	in_file.readline()
	for i in range(test_num): 
		f = in_file.readline().replace('\n','').split(',')  
		split = f[1].split(" ")  
		feature = [float(x)/255 for x in split] 
		test_in.append(feature) 
test_in = np.array(test_in)  
test_in = test_in.reshape(test_in.shape[0],48,48,1)
model0 = load_model('model_rotate.h5')
model1 = load_model('model1.h5')
#model2 = load_model('model2.h5')
#model3 = load_model('model3.h5')
#model4 = load_model('model4.h5') 
model5 = load_model('model5.h5') 
result0 = model0.predict(test_in)
result1 = model1.predict(test_in)
#result2 = model2.predict(test_in)
#result3 = model3.predict(test_in)
#result4 = model4.predict(test_in)
result5 = model5.predict(test_in)
result = result5 +  result0 + result1 
with open(sys.argv[2], 'w') as fp:
    fp.write('id,label\n')
    for i in range(test_num): 
        max_ = 0
        ans = -10
        for j in range(7):
            if result[i][j] > max_:
                max_ = result[i][j]
                ans = j
        fp.write('%d,%d\n' % (i, ans))