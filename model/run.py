from data_processing import clean_data, agg_description
from fs_TFIDF import tfidf_matrix_features, find_top_features_per_wine
from fs_LDA import latentdirichletallocation
from rec_CosSim import RecCosineSimilarity

from fs_TFIDF import _lemmatize_tokens_pos
import string
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

filepath = '../../data/sample.csv' # sample dataset for build-testing
# filepath = '../../data/all_wine_data.csv' # full dataset

'''Regular clean_data'''
wine_df, wine_stop_lib = clean_data(filepath)
descriptions = wine_df['description'][:500] #Test with 500

# Aggregate Descriptions
# agg_df, groups = agg_description(wine_df)
# descriptions = agg_df.values

#TFIDF (Regular Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib)

#TFIDF (Lemmatize Tokens - with POS filter)
vect, tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=1)

#TFIDF (Porter Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=2)

#TFIDF (Snowball Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=3)

#Recommendation
def make_recommendation(rec_for_id, vect, tfidf_docs, features, n_size=5):
    cs = RecCosineSimilarity(n_size)
    cs.fit(vect, tfidf_docs)
    test_one = cs.recommend_to_one(wine_id = rec_for_id)

    top_feats, top_idx = find_top_features_per_wine(features, tfidf_docs, n_features=5)

    print "Wine I like: {}".format(wine_df['product'][rec_for_id])
    print " Features: {}".format(top_feats[rec_for_id])
    print "Recommendations: "
    for n, idx in enumerate(test_one):
        print "{}. {}".format(n+1, wine_df['product'][idx])
        print "       {}".format(top_feats[idx])

rec_for_id = 333 # Want recommendations similar to this ID
# make_recommendation(rec_for_id, vect, tfidf_docs, features)

# Recommendation from User Input

def make_rec_from_user(vect, tfidf_docs, n_size=5):
    cs = RecCosineSimilarity(n_size)
    cs.fit(vect, tfidf_docs)
    user_input = raw_input("Describe the flavors you like in wine: ")
    test_one = cs.recommend_user_input(user_input)
    print "Recommendations: "
    for n, idx in enumerate(test_one):
        print "{}. {}".format(n+1, wine_df['product'][idx])
        print "    {}, {}".format(wine_df['country'][idx], wine_df['price'][idx])

make_rec_from_user(vect, tfidf_docs)

'''
#Recommendation for Aggregate Descriptions
group_id = 500 # Want recommendations similar to this group ID

cs = RecCosineSimilarity(n_size=5)
cs.fit(tfidf_docs)
test_one = cs.recommend_to_one(wine_id = group_id)

print "Wine I like: {}".format(agg_df.index[group_id])
print "Recommendations: "
for n, idx in enumerate(test_one):
    akey = agg_df.index[idx]
    print "{}. {}".format(n+1, akey)
    wine_idx = groups[akey].tolist()
    print "  Wine name(s): "
    for i in wine_idx:
        print "   {}".format(wine_df['product'][i])
        #print "   {}".format(wine_df.values[i])
# LDA
cv_docs, features, lda = latentdirichletallocation(descriptions, wine_stop_lib, num_topics=9, passes=20)
lda_map = lda.fit_transform(cv_docs) # document by topic matrix
'''

#
# import numpy as np
#
#
#
# cs = RecCosineSimilarity(5)
# cs.fit(tfidf_docs)
#
# user_input = "Grippy leather and tobacco provide a gravelly texture and burly   \
#               nature to this medium-bodied wine, which finds its groove in      \
#               the glass. Concentrated and dense, it offers smoother elements    \
#               of black currant, cassis and leather on the finish."
#
#
#
#
# user_lem = _lemmatize_tokens_pos([user_input])
# user_matrix = vect.transform(user_lem) #returns matrix 1 x # features
# user_cs = cosine_similarity(user_matrix, tfidf_docs)
# rec_ids = np.argsort(user_cs[0])[-5:][::-1] # wine ids with best match
