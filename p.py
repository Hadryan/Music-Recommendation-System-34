import pandas as pd
import numpy as np
from numpy import int64

import IPython.display as Disp

import sklearn
from sklearn.decomposition import TruncatedSVD

import csv
####        Read details related to songs
song_data = pd.read_csv(r"C:\Users\Shreya Garg\Desktop\Shreya Garg_5th_E_2014866\song_data.csv") 
print(song_data.head(3))
print('\n')

print(song_data.describe())
print('\n')

print(song_data.groupby("artist_name")["song_id"].count().sort_values(ascending = False))
print('\n')

####        Read dataset that shows how many times a user plays each song into pandas dataframe
trpilets_file = pd.read_csv(r"C:\Users\Shreya Garg\Desktop\Shreya Garg_5th_E_2014866\triplets_file.csv")
print(trpilets_file.head())
print('\n')

print(trpilets_file.groupby('user_id')['listen_count'].count().sort_values(ascending = False))
print('\n')

####        Merge songs and user dataset into one table
combined_songs = pd.merge(trpilets_file, song_data.drop_duplicates())
print(combined_songs.head())

####        Popular songs
combined_songs.groupby('song_id')['listen_count'].count().sort_values(ascending = False) #group song id and rating
combined_songs.groupby('title')['listen_count'].count().sort_values(ascending=False) 
# finding most popular song name
song_title_df = pd.DataFrame({'count': combined_songs.groupby(['title']).size()}).reset_index()
song_title_df.columns = ['title', 'count']
print(song_title_df[(song_title_df['count'] > 3000)  & (song_title_df['count']<3113) ].head())
print('\n')
#can be used for finding fav artist
print(combined_songs.groupby('artist_name')['listen_count'].count().sort_values(ascending = False))
ct_df = combined_songs.pivot_table(values = 'listen_count', index = 'user_id', columns = 'title', fill_value = 0)
print(ct_df.head())

x = ct_df.values.T
x.shape

svd  = TruncatedSVD(n_components=20, random_state=17)
result_matrix = svd.fit_transform(x)
print(result_matrix.shape)

corr_mat = np.corrcoef(result_matrix)
print(corr_mat.shape)

####        Print books related to song list
song_names = ct_df.columns
song_list = list(song_names)
print(song_list)

query_index = song_list.index("Heartbreak Warfare")
print(query_index)

corr_similar_songs = corr_mat[query_index]
corr_similar_songs.shape
print(corr_similar_songs)
print(type(song_list))
print((corr_similar_songs < 1.0) & (corr_similar_songs > 0.9))

print(list(song_names[(corr_similar_songs < 1.0) & (corr_similar_songs > 0.98)]))