from data_processing import clean_data
from fs_TFIDF import tfidf_matrix_features
from rec_CosSim import RecCosineSimilarity

filepath = '../csv/sample.csv'
wine_df, wine_stop_lib = clean_data(filepath)
descriptions = wine_df['description']

#TFIDF (Regular Tokens)
tfidf_docs, features = tfidf_matrix_features(descriptions, wine_stop_lib)

rec_for_id = 100 # Want recommendations similar to this ID

cs = RecCosineSimilarity(n_size=5)
cs.fit(tfidf_docs)
test_one = cs.recommend_to_one(wine_id = rec_for_id)

print "Wine I like: {}".format(wine_df['product'][rec_for_id])
print "Recommendations: "
for n, idx in enumerate(test_one):
    print "{}. {}".format(n+1, wine_df['product'][idx])
