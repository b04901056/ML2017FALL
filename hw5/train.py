import numpy as np 
import keras
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint  
import os
from keras.models import load_model
import sys
import matplotlib.pyplot as plt  

avg = 3.58171208604
std = 1.11689766115

def unison_shuffled_copies(a, b ,c): 
    p = np.random.permutation(len(a))
    return a[p], b[p] , c[p]

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

import numpy as np      
import sys

user = np.load('user.npy')
movie = np.load('movie.npy')
ranking = np.load('ranking.npy') 
user_num = 6041
movie_num = 3952
user , movie , ranking = unison_shuffled_copies(user,movie,ranking)


user_id = np.load('user_id.npy').tolist()
user_gender = np.load('user_gender.npy').tolist()
user_age = np.load('user_age.npy').tolist() 
movie_id = np.load('movie_id.npy').tolist()
movie_genre = np.load('movie_genre.npy').tolist() 


user_age = (user_age-np.mean(user_age))/np.std(user_age)

all_genres = np.array([])
for genres in movie_genre:
    for genre in genres.split('|'):
        all_genres = np.append(all_genres, genre)
all_genres = np.unique(all_genres)

movies_info = np.zeros((movie_num, all_genres.shape[0]))
users_info = np.zeros((user_num, 2))

for idx, u_id in enumerate(user_id): 
    gender = 1 if user_gender == 'M' else 0 
    tmp = [gender, user_age[idx]] 
    users_info[u_id - 1] = tmp
print(users_info)
for idx, mo_id in enumerate(movie_id):
    genres = movie_genre[idx].split('|')
    tmp = np.zeros(all_genres.shape[0]) 
    for genre in genres:
        tmp[np.where(all_genres == genre)[0][0]] = 1
    movies_info[mo_id - 1] = tmp
print(movies_info) 

############################ build model ############################

print('build model...') 

embedding_dim = 256
DROPOUT_RATE = 0.3

User_input = keras.layers.Input(shape = (1,),dtype = 'int32')
User_embed = keras.layers.Embedding(output_dim = embedding_dim,
                       input_dim = user_num,
                       input_length = 1,
                       embeddings_initializer='random_normal',
                       embeddings_regularizer = keras.regularizers.l2(1e-5),
                       trainable=True)(User_input)
User_reshape = keras.layers.Reshape((embedding_dim,))(User_embed)
User_bias = (keras.layers.Embedding(output_dim = 1,
                       input_dim = user_num,
                       input_length = 1,
                       embeddings_initializer='zeros',
                       embeddings_regularizer = keras.regularizers.l2(1e-5),
                       trainable=True)(User_input))
User_bias = keras.layers.Flatten()(User_bias)

Movie_input = keras.layers.Input(shape = (1,),dtype = 'int32')
Movie_embed = keras.layers.Embedding(output_dim = embedding_dim,
                        input_dim = movie_num,
                        input_length = 1,
                        embeddings_initializer='random_normal',
                        embeddings_regularizer = keras.regularizers.l2(1e-5),
                        trainable=True)(Movie_input)
Movie_reshape = keras.layers.Reshape((embedding_dim,))(Movie_embed)
Movie_bias = (keras.layers.Embedding(output_dim = 1,
                        input_dim = movie_num,
                        input_length = 1,
                        embeddings_initializer='zeros',
                        embeddings_regularizer = keras.regularizers.l2(1e-5),
                        trainable=True)(Movie_input))
Movie_bias = keras.layers.Flatten()(Movie_bias)

if(sys.argv[1]=='feature'):
    print('Use dnn model with extra feature')
    U_info_emb = keras.layers.Embedding(input_dim = user_num,output_dim = users_info.shape[1],weights=[users_info],trainable=True)(User_input)
    U_info_emb = keras.layers.Flatten()(U_info_emb)
    M_info_emb = keras.layers.Embedding(input_dim = movie_num,output_dim = movies_info.shape[1],weights=[movies_info],trainable=True)(Movie_input)
    M_info_emb = keras.layers.Flatten()(M_info_emb)

    concat = keras.layers.Concatenate()([User_reshape, Movie_reshape, U_info_emb, M_info_emb])
    dnn = keras.layers.Dense(256, activation='relu')(concat)
    dnn = keras.layers.Dropout(DROPOUT_RATE)(dnn)
    dnn = keras.layers.BatchNormalization()(dnn) 
    output = keras.layers.Dense(1, activation='relu')(dnn)
    model = keras.models.Model(inputs=[User_input, Movie_input], outputs = output)
    model.compile( loss='mean_squared_error', optimizer='adam' )
    model.summary()  
    train_history = model.fit([user, movie ], ranking, batch_size=10000, epochs=200 , validation_split=0.1 )   
    model.save('model_feature.h5')
    show_train_history( train_history , 'loss' , 'val_loss' )
     
if(sys.argv[1]=='mf'):
    user_input = keras.layers.Input(shape=[1])  
    user_vec = keras.layers.Flatten()(keras.layers.Embedding(user_num, 256,
                        trainable=True)(user_input))
    user_vec = keras.layers.Dropout(0.5)(user_vec)  

    movie_input = keras.layers.Input(shape=[1])  
    movie_vec = keras.layers.Flatten()(keras.layers.Embedding(movie_num, 256,
                        trainable=True)(movie_input))
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)  
 
    input_vecs = keras.layers.dot([movie_vec, user_vec], axes= -1 ) 

    model = keras.models.Model([user_input, movie_input], input_vecs)
    model.compile( loss='mean_squared_error', optimizer='adam' )
    model.summary() 

    train_history = model.fit([user, movie ], ranking, batch_size=10000, epochs=100 , validation_split=0.1 )   
    model.save('model_mf_256.h5')
    show_train_history( train_history , 'loss' , 'val_loss' )

if(sys.argv[1]=='dnn'):
    
    concat = keras.layers.Concatenate()([User_reshape, Movie_reshape])
    dnn = keras.layers.Dense(256, activation='relu')(concat)
    dnn = keras.layers.Dropout(DROPOUT_RATE)(dnn)
    dnn = keras.layers.BatchNormalization()(dnn)
    dnn = keras.layers.Dense(256, activation='relu')(dnn)
    dnn = keras.layers.Dropout(DROPOUT_RATE)(dnn)
    dnn = keras.layers.BatchNormalization()(dnn)
    dnn = keras.layers.Dense(256, activation='relu')(dnn)
    dnn = keras.layers.Dropout(DROPOUT_RATE)(dnn)
    dnn = keras.layers.BatchNormalization()(dnn)
    output = keras.layers.Dense(1, activation='relu')(dnn)
    model = keras.models.Model(inputs=[User_input, Movie_input], outputs = output) 
    
    model.compile( loss='mean_squared_error', optimizer='adam' )
    model.summary()  
    train_history = model.fit([user, movie ], ranking, batch_size=10000, epochs=200 , validation_split=0.1 )   
    model.save('model_dnn.h5')
    show_train_history( train_history , 'loss' , 'val_loss' )
    

if(sys.argv[1]=='bias'):
      
    user_input = keras.layers.Input(shape=[1])  
    user_vec = keras.layers.Flatten()(keras.layers.Embedding(user_num, 256)(user_input))
    user_vec = keras.layers.Dropout(0.5)(user_vec)  

    movie_input = keras.layers.Input(shape=[1])  
    movie_vec = keras.layers.Flatten()(keras.layers.Embedding(movie_num, 256)(movie_input))
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)

    User_bias = (keras.layers.Embedding(user_num, 1 ,trainable=True)(user_input))
    User_bias = keras.layers.Flatten()(User_bias)
    Movie_bias = (keras.layers.Embedding(movie_num, 1 ,trainable=True)(movie_input))
    Movie_bias = keras.layers.Flatten()(Movie_bias)  
 
    input_vecs = keras.layers.dot([movie_vec, user_vec], axes= -1 ) 

    if (sys.argv[2]=='y') :
        if (sys.argv[3]=='y'):
            print('Use matrix factorization with movie and user bias')
            Main_add = keras.layers.Add()([input_vecs, Movie_bias, User_bias])
        else:
            Main_add = keras.layers.Add()([input_vecs, Movie_bias])
            print('Use matrix factorization with movie bias')
    else:
        if (sys.argv[3]=='y'):
            print('Use matrix factorization with user bias')
            Main_add = keras.layers.Add()([input_vecs, User_bias])
        else:
            Main_add = input_vecs
            print('Use matrix factorization without bias')

    model = keras.models.Model([user_input, movie_input], Main_add)
    model.compile( loss='mean_squared_error', optimizer='adam' )
    model.summary() 

    train_history = model.fit([user, movie ], ranking, batch_size=10000, epochs=200 , validation_split=0.1 )   
    model.save('model_bias.h5')
    show_train_history( train_history , 'loss' , 'val_loss' )