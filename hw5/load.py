import numpy as np
import sys

print('processing training data ...')

movie = []
user = []
ranking = []

with open(sys.argv[1], 'r', encoding='UTF-8') as fp:
	fp.readline()
	for i in range (899873):
		a = fp.readline().replace('\n','').split(',')
		user.append(int(a[1]))
		movie.append(int(a[2]))
		ranking.append(int(a[3]))

user = np.array(user)
movie = np.array(movie)

avg = np.mean(ranking)
std = np.std(ranking)

print('avg=',avg)
print('std=',std)

ranking = ( ranking - avg ) / std
ranking = np.array(ranking)

np.save('movie',movie)
np.save('user',user)
np.save('ranking',ranking)

#################################################################################

print('processing user data ...')

user_id = []
user_gender = []
user_age = []
 
 
with open(sys.argv[2], 'r', encoding='UTF-8') as fp:
	fp.readline()
	for i in range (6040):
		a = fp.readline().replace('\n','').split('::')
		user_id.append(int(a[0]))
		user_gender.append(a[1])
		user_age.append(int(a[2]))

user_id = np.array(user_id)
user_gender = np.array(user_gender) 
user_age = np.array(user_age) 

np.save('user_id',user_id)
np.save('user_gender',user_gender)
np.save('user_age',user_age)

#################################################################################

print('processing movie data ...')

movie_id = []
movie_genre = []

with open(sys.argv[3], 'r', encoding='ISO-8859-1') as fp:
	fp.readline()
	for i in range (3883):
		a = fp.readline().replace('\n','').split('::')
		movie_id.append(int(a[0]))
		movie_genre.append(a[2]) 

movie_id = np.array(movie_id)
movie_genre = np.array(movie_genre)
 

np.save('movie_id',movie_id)
np.save('movie_genre',movie_genre) 
 