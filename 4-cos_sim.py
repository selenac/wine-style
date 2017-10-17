import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

filename = '../eda/train_sample.csv'

def load_data(filename):
    '''
    input: file csv format
    output: pandas dataframe
    '''
    df = pd.read_csv(filename)
    return df

def create_variety_list(df):
    '''
    unique variety list to add to stop words
    '''
    return map(str.lower, df['variety'].unique())

def create_wine_stop(df):
    '''
    combine reg english stops and wine specific text
    return wine stop words to use in vectorizing
    '''
    # working list of stop words
    wine_stop_lib = ['aromas', 'drink', 'fruit', 'palate','wine', 'like', 'bit']
    return stopwords.words('english') + wine_stop_lib + create_variety_list(df)

def get_tfidf_vect(df):
    wine_stop_words = create_wine_stop(df)
    vectorizer = TfidfVectorizer(stop_words = wine_stop_words,
                                decode_error = 'ignore',
                                strip_accents = 'unicode',
                                max_df = 0.97,
                                min_df = 0.03,
                                ngram_range = (1,2),
                                lowercase = True)
    return vectorizer

def transform_vect(df):
    v = get_tfidf_vect(df)
    tfidf = v.fit_transform(df['description'])
    return tfidf

def cosine_sim_matrix(df):
    tfidf = transform_vect(df)
    cosine_similiarity = linear_kernel(tfidf, tfidf)
    return cosine_similiarity

######################################
wine = load_data(filename)
cs = cosine_sim_matrix(wine)

tfidf = transform_vect(wine)
tfidf

cs.shape

fifty = wine['variety'][0:50]

for i , doc1 in enumerate(fifty):
    for j, doc2 in enumerate(fifty):
        if i != j and cs[i,j] > 0.25:
            print i, doc1, j, doc2, cs[i, j]


######################################
stop = stopwords.words('english')
wine_stop_words = set(stop + ['aromas', 'cabernet', 'drink', 'fruit', 'palate', 'pinot',
                    'sauvignon', 'wine', 'like', 'bit', 'chardonnay'])

vect = TfidfVectorizer(stop_words = wine_stop_words,
                       decode_error = 'ignore',
                       strip_accents = 'unicode',
                       max_df = 0.97,
                       min_df = 0.03,
                       ngram_range = (1,2),
                       lowercase = True)

desc_vectors = vect.fit_transform(wine['description'])
desc_vectors.shape
vect.get_feature_names()

print desc_vectors[20]

cosine_similarities = linear_kernel(desc_vectors[0:1], desc_vectors).flatten()
related_docs_indices = cosine_similarities.argsort()[:-10:-1]
related_docs_indices
cosine_similarities[related_docs_indices]
wine.info()
for i in related_docs_indices:
    print wine['description'][i], wine['variety'][i], wine['country'][i], wine['winery'][i]


wine.values[0]

def comp_description(query, results_number=20):
        results=[]
        q_vector = vect.transform([query])
        print("Comparable Description: ", query)
        results.append(cosine_similarity(q_vector, desc_vectors.toarray()))
        f=0
        elem_list=[]
        for i in results[:10]:
            for elem in i[0]:
                    #print("Review",f, "Similarity: ", elem)
                    elem_list.append(elem)
                    f+=1
            print("The Review Most similar to the Comparable Description is Description #" ,elem_list.index(max(elem_list)))
            print("Similarity: ", max(elem_list))
            if sum(elem_list) / len(elem_list)==0.0:
                print("No similar descriptions")
            else:
                print(wine['description'].loc[elem_list.index(max(elem_list)):elem_list.index(max(elem_list))])
                print(wine['variety'][elem_list.index(max(elem_list))])


comp_description('A semi-dry white wine with pear, citrus, and tropical fruit flavors; crisp and refreshing.')
