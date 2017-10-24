from data_processing import clean_data, agg_description
from fs_TFIDF import tfidf_matrix_features, find_top_features_per_wine
from fs_LDA import latentdirichletallocation
from rec_CosSim import RecCosineSimilarity


filepath = '../../data/sample.csv' # sample dataset for build-testing
# filepath = '../../data/wine_data.csv' # full dataset

'''Regular clean_data'''
wine_df, wine_stop_lib = clean_data(filepath)
descriptions = wine_df['description']

# Aggregate Descriptions
# agg_df, groups = agg_description(wine_df)
# descriptions = agg_df.values

#TFIDF (Regular Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib)

#TFIDF (Lemmatize Tokens)
tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=1)

#TFIDF (Porter Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=2)

#TFIDF (Snowball Stem Tokens)
# tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib, stemlem=3)


#Recommendation
rec_for_id = 560 # Want recommendations similar to this ID

cs = RecCosineSimilarity(n_size=5)
cs.fit(tfidf_docs)
test_one = cs.recommend_to_one(wine_id = rec_for_id)

top_feats, top_idx = find_top_features_per_wine(features, tfidf_docs)

print "Wine I like: {}".format(wine_df['product'][rec_for_id])
print " Features: {}".format(top_feats[rec_for_id])
print "Recommendations: "
for n, idx in enumerate(test_one):
    print "{}. {}".format(n+1, wine_df['product'][idx])
    print "       {}".format(top_feats[idx])

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
