import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecCosineSimilarity(object):

    def __init__(self, n_size):
        self.n_size = n_size

    def fit(self, wine_mat):
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

    def recommendation_matrix():
        '''
        Output: cs_matrix - numpy array reference for cs for wine x wine
        '''
        cs_matrix = cosine_similarity(self.wine_mat)
        return cs_matrix

if __name__ == '__main__':
    pass
