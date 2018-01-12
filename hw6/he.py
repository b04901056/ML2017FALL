from skimage.io import imread, imsave
import os
import numpy as np 
import sys 
from skimage import transform
 

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype(np.uint8)
    return M

IMAGE_PATH = sys.argv[1]+'/'
filelist = os.listdir(IMAGE_PATH)
img_shape = imread(IMAGE_PATH+filelist[0]).shape 

img_data = []
for i, filename in enumerate(filelist):
    tmp = imread(IMAGE_PATH+filename)  
    img_data.append(tmp.flatten())
img_data = np.array(img_data)

X_mean = np.mean(img_data, axis=0) 
x = img_data - X_mean 
u, s, v = np.linalg.svd(x.T, full_matrices=False) 
eigonvector = u.T 
 

#print('reconstruct ...')


picked_img = imread(sys.argv[2])  
X = picked_img.flatten()
X -= X_mean
weight = np.array([X.T.dot(eigonvector[i]) for i in range(415)]) 
reconstruct = process(u[:,:4].dot(weight[:4].T)+ X_mean.astype(np.uint8))
imsave('reconstruction.jpg', reconstruct.reshape(img_shape), quality = 80)  