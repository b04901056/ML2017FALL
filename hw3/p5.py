from keras import backend as K
from keras.models import load_model
import os
import numpy  
from matplotlib import pyplot as plt
 
test_in = numpy.load('x_train.npy')
test_in=test_in.reshape(test_in.shape[0],48,48,1)
 
filter_index = 0   

fig = plt.figure(figsize=(10,3))  

model = load_model('model5.h5')
input_img = model.input
con_model = K.function([input_img, K.learning_phase()], [model.layers[3].output])

layer_output = model.layers[3].output


NUM = 13
for filter_index in range(32):

    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])
    output = numpy.random.random((1, 48, 48, 1)) 
 

    tmp = test_in[NUM].reshape((1,48, 48, 1)) 
    output = con_model([tmp, 0])[0]

    ax = fig.add_subplot(32/16,16,filter_index + 1) 
    #ax.imshow(test_in[NUM].reshape((48, 48)), cmap = 'gray')
    ax.imshow(output[0, :, :, filter_index],cmap='Greens')
    plt.xticks(numpy.array([]))
    plt.yticks(numpy.array([]))

#fig.suptitle('Filters of layer %s (# Ascent Epoch 200 )' % (model.layers[3].name))
fig.suptitle('Output of layer %s (Given image %d)' % (model.layers[3].name, NUM))
fig.savefig('13.png')