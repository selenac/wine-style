import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fs_TFIDF import _lemmatize_tokens_pos

class RecCosineSimilarity(object):

    def __init__(self, n_size):
        self.n_size = n_size

    def fit(self, vect, wine_mat):
        self.vect = vect
        self.wine_mat = wine_mat
        self.n_wines = wine_mat.shape[0]

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

    def recommendation_matrix(self):
        '''
        Output: cs_matrix - numpy array reference for cs for wine x wine
        '''
        cs_matrix = cosine_similarity(self.wine_mat)
        return cs_matrix

    def recommend_user_input(self, user_input):
        '''
        Input:
            user_input - string description from user
            vect - fitted vectorizer
        Output: rec_ids - numpy array of n_size with top cosine similar wines
        '''
        user_lem = _lemmatize_tokens_pos([user_input])
        user_matrix = self.vect.transform(user_lem) #returns matrix 1 x # features
        user_cs = cosine_similarity(user_matrix, self.wine_mat)
        rec_ids = np.argsort(user_cs[0])[-self.n_size:][::-1] # wine ids with best match
        return rec_ids

if __name__ == '__main__':
    pass
