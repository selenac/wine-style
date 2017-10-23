from data_processing import clean_data
from fs_TFIDF import tfidf_matrix_features
from fs_LDA import latentdirichletallocation
from rec_CosSim import RecCosineSimilarity

filepath = '../../data/sample.csv' # sample dataset for build-testing
# filepath = '../../data/wine_data.csv' # full dataset

'''Regular clean_data'''
# wine_df, wine_stop_lib = clean_data(filepath)
# descriptions = wine_df['description']

# Aggregate Descriptions
wine_df, wine_stop_lib, groups = clean_data(filepath, agg_desc=1)
descriptions = wine_df.values

#TFIDF (Regular Tokens)
tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib)

#TFIDF (Lemmatize Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=1)

#TFIDF (Porter Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=2)

#TFIDF (Snowball Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=3)

'''
#Recommendation
rec_for_id = 100 # Want recommendations similar to this ID

cs = RecCosineSimilarity(n_size=5)
cs.fit(tfidf_docs)
test_one = cs.recommend_to_one(wine_id = rec_for_id)

print "Wine I like: {}".format(wine_df['product'][rec_for_id])
print "Recommendations: "
for n, idx in enumerate(test_one):
    print "{}. {}".format(n+1, wine_df['product'][idx])
'''

#Recommendation for Aggregate Descriptions
rec_for_id = 100 # Want recommendations similar to this group ID

cs = RecCosineSimilarity(n_size=5)
cs.fit(tfidf_docs)
test_one = cs.recommend_to_one(wine_id = rec_for_id)

print "Wine I like: {}".format(wine_df.index[rec_for_id])
print "Recommendations: "
for n, idx in enumerate(test_one):
    print "{}. {}".format(n+1, wine_df.index[idx])


'''
# LDA
cv_docs, features, lda = latentdirichletallocation(descriptions, wine_stop_lib, num_topics=9, passes=20)
lda_map = lda.fit_transform(cv_docs)
'''
