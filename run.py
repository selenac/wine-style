from data_processing import clean_data
from fs_TFIDF import tfidf_matrix_features

filepath = '../csv/sample.csv'
wine_df, wine_stop_lib = clean_data(filepath)

tfidf_docs, features, top = tfidf_matrix_features(wine_df['description'], wine_stop_lib)
