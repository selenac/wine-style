'''
Module containing model fitting code for Wine Style web application.
Implements data processing, TFIDF of review text, and performs
cosine similarity between wine descriptions to find recommendations
with a similar taste description.
'''

from model.data_processing import clean_data
from model.fs_TFIDF import tfidf_matrix_features, _lemmatize_tokens_pos

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class RecCosineSimilarity(object):

    def __init__(self, n_size=5):
        self.n_size = n_size

    def fit(self, vect, wine_mat):
        self.vect = vect
        self.wine_mat = wine_mat
        # self.n_wines = wine_mat.shape[0]

    def recommendation_matrix(self):
        '''
        Output: cs_matrix - numpy array reference for cs for wine x wine
        '''
        cs_matrix = cosine_similarity(self.wine_mat)
        return cs_matrix

    def recommend_to_one(self, wine_id):
        '''
        Input: wine_id - index for wine seeking recommendations
        Output: rec_ids - numpy array of n_size with top cosine similar wines
        '''
        wine_vec = self.wine_mat[wine_id].toarray()
        cs = cosine_similarity(wine_vec, self.wine_mat)
        rec_ids = np.argsort(cs[0])[-(self.n_size+1):][::-1]
        rec_ids = rec_ids[1:] #drop 1st element (always equal to wine_id)
        return rec_ids

    def recommend_user_input(self, user_input):
        '''
        Input:
            user_input - string description from user
        Output: rec_ids - numpy array of n_size with top cosine similar wines
        '''
        user_lem = _lemmatize_tokens_pos([user_input])
        user_matrix = self.vect.transform(user_lem) #returns matrix 1 x # features
        user_cs = cosine_similarity(user_matrix, self.wine_mat)
        rec_ids = np.argsort(user_cs[0])[-self.n_size:][::-1] # wine ids with best match
        return rec_ids

if __name__ == '__main__':
    filepath = '../../data/sample.csv' # sample dataset for build-testing
    # filepath = '../../data/all_wine_data.csv' # full dataset

    wine_df, stop_lib = clean_data(filepath)
    wine_df.to_pickle('../static/wine_df.pkl') # lookup for wine_id and product list

    X = wine_df['description']
    # Create TfidfVectorizer and Lemmatize tokens with WordNetLemmatizer + POS tagging
    vectorizer, wine_matrix, features = tfidf_matrix_features(X, stop_lib, stemlem=1)


    n_recs = 5
    cs = RecCosineSimilarity(n_recs)
    cs.fit(vectorizer, wine_matrix)
    with open ('../static/cos_sim_model.pkl', 'wb') as f:
        pickle.dump(cs, f)
