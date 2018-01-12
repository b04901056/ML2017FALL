import numpy as np
import sys  
from keras.models import load_model 
from sklearn import cluster, datasets  
 
train_num = 130000
sample_num = 1980000
X = np.load(sys.argv[1])
X = X.astype('float32') / 255.  
X = np.reshape(X, (len(X), -1))

encoder = load_model('encoder.h5')

encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1) 
kmeans_fit = cluster.KMeans( n_clusters=2).fit(encoded_imgs)

test_first = []
test_second = []
  
with open(sys.argv[2], 'r', encoding='UTF-8') as fp:
	fp.readline()
	for i in range(sample_num): 
		a = fp.readline().replace('\n','').split(',')
		test_first.append(int(a[1])) 
		test_second.append(int(a[2]))
  
with open(sys.argv[3], 'w') as fp:
	fp.write('ID,Ans\n')
	for i in range(sample_num):
		ans = 0 
		if kmeans_fit.labels_[test_first[i]] == kmeans_fit.labels_[test_second[i]]:
			ans = 1 
		fp.write('%d,%d\n' % (i, ans))  