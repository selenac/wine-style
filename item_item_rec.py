import numpy as np
import pandas as pd
from scipy import sparse

# import cos_sim.py functions

class ItemItemRecommender(object):

    def __init__(self, n_size):
        self.n_size = n_size

    def fit(self, cos_sim):
        self.cos_sim = cos_sim
        self.n_items = cos_sim.shape[0]
        self._set_nsize()

    def _set_nsize(self):
        least_to_most_sim_indexes = np.argsort(self.cos_sim, 1)
        self.n = least_to_most_sim_indexes[:, -self.n_size:]

    def pred_one(self):
        pass

    def top_n_recs(self, item_index):
        output = []
        arr = cs[item_id].argsort()[-(self.n_size+1):][::-1]
        arr = arr[1:] #drop 1st element (always equal to item_id)
        for a in arr:
            output.append((a, self.cos_sim[item_index][a]))
        return output

#########################################

if __name__ == '__main__':
