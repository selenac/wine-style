import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

''' Loading Data into a dataframe '''

def load_data(filename):
    '''
    input: file csv format
    output: pandas dataframe
    '''
    df = pd.read_csv(filename)
    return df

''' Update dataframe with product name column for identification '''

def create_product_name(df):
    '''
    create wine product column with name
    '''
    df['product'] = df['winery'] + ' ' + df['designation'].fillna('') + ' ' + df['variety']
    return df

''' Additional stop words generated related to wine '''

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
    wine_stop_lib = ['aromas', 'drink', 'fruit', 'palate','wine', 'like', 'bit',
                     'flavor', 'fine', 'sense', 'note', 'notes', 'frame']
    return stopwords.words('english') + wine_stop_lib + create_variety_list(df)

''' Tokenizers '''

def lemmatize_descriptions(descriptions):
    l = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(l.lemmatize(w) for w in d.split())
    return [lemmatize(desc.decode(errors='ignore')) for desc in descriptions]

def snowball_stem_descriptions(descriptions):
    pass

''' Vectorizers '''

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

def transform_vect(df, lem=False):
    v = get_tfidf_vect(df)
    desc = df['description']
    if lem:
        desc = lemmatize_descriptions(desc)
    tfidf = v.fit_transform(desc)
    return tfidf

# def get_count_vect(df):
#     wine_stop_words = create_wine_stop(df)
#     vectorizer = CountVectorizer(stop_words = wine_stop_words,
#                                 decode_error = 'ignore',
#                                 strip_accents = 'unicode',
#                                 # max_df = 0.97,
#                                 # min_df = 0.03,
#                                 # ngram_range = (1,2),
#                                 lowercase = True)
#     return vectorizer
#
# def transform_countvect(df, lem=False):
#     v = get_count_vect(df)
#     desc = df['description']
#     if lem:
#         desc = lemmatize_descriptions(desc)
#     cv = v.fit_transform(desc)
#     return cv

''' Similarity Comparisons '''

def cosine_sim_matrix(df, tf=True, lem=False):
    if tf:
        tfidf = transform_vect(df, lem)
        cosine_similiarity = linear_kernel(tfidf, tfidf)
    else:
        cv = transform_countvect(df, lem)
        cosine_similiarity = linear_kernel(cv, cv)
    return cosine_similiarity

''' Recommendations'''

def pred_one(cs, item_id):
    arr = cs[item_id].argsort()[-2:][::-1]
    arr = arr[1:]
    return (arr[0], cs[item_id][arr[0]])

def top_n_sim(cs, item_id, n=5):
    '''
    input:
        cosine_similarity matrix (np)
        item_id: wine id
        n: number of similar wines
    output:
        w_list: list of n tuples (wine_ids similar to item_id, similarity score)
    '''
    output = []
    arr = cs[item_id].argsort()[-(n+1):][::-1]
    arr = arr[1:] #drop 1st element (always equal to item_id)
    for a in arr:
        output.append((a, cs[item_id][a]))
    return output

def return_recs_name(df, cs, item_id, n=5):
    '''
    '''
    recommendation = []
    rec_ids = top_n_sim(cs, item_id, n)
    for rec_id in rec_ids:
        recommendation.append(df['product'][rec_id[0]])
    return recommendation

def return_recs_df(df, cs, item_id, n=5):
    '''
    '''
    recommendation = []
    rec_ids = top_n_sim(cs, item_id, n)
    for rec_id in rec_ids:
        recommendation.append(df.values[rec_id[0]])
    return recommendation

def rec_comparison_csv(df, n=5):
    '''Compare Different Recs'''
    info = []
    columns = ['wine_id', 'wine_selected', 'token_type',
                'rec_rank', 'sim_score', 'rec_wine_id', 'rec_wine_name']
    check_wine_ids = np.random.choice(len(wine), n) #random wines to check

    cs = cosine_sim_matrix(df, tf=True, lem=False)
    csl = cosine_sim_matrix(df, tf=True, lem=True)

    for wine_id in check_wine_ids:
        sim_wines = top_n_sim(cs, wine_id)
        l_sim_wines = top_n_sim(csl, wine_id)
        for i in xrange(len(sim_wines)):
            score = sim_wines[i][1]
            rec_wine_id = sim_wines[i][0]
            info.append([wine_id, df['product'][wine_id], 'reg', i+1,
            score, rec_wine_id, df['product'][rec_wine_id]])
        for i in xrange(len(l_sim_wines)):
            score = l_sim_wines[i][1]
            rec_wine_id = l_sim_wines[i][0]
            info.append([wine_id, df['product'][wine_id],'lem', i+1,
            score, rec_wine_id, df['product'][rec_wine_id]])

    comp_df = pd.DataFrame(info, columns=columns)
    comp_df.to_csv('../csv/rec_comparison.csv')
    print 'Done'

######################################

if __name__ == '__main__':

    filename = '../csv/sample.csv'
    wine = load_data(filename)
    wine = create_product_name(wine)
    rec_comparison_csv(wine)

    #Regular tokenizing
    #cs = cosine_sim_matrix(df=wine, tf=True, lem=False)
    #Lemmatize the descriptions
    #csl = cosine_sim_matrix(df=wine, tf=True, lem=True)

    # Test recommendations
    # check_wine_id = 28
    # sim_wines = top_n_sim(cs, check_wine_id)
    # l_sim_wines = top_n_sim(csl, check_wine_id)
    # print 'Finding similar wines to {}'.format(wine['product'][check_wine_id])
    # print 'Description: {}'.format(wine['description'][check_wine_id])
    # print
    # print 'Regular: {}'.format(sim_wines)
    # print 'Lem:     {}'.format(l_sim_wines)
    # print
    # print return_recs_df(wine, cs, check_wine_id)
    # print
    # print return_recs_df(wine, csl, check_wine_id)

#TODO get top features from each tokenized description



    # cvcs = cosine_sim_matrix(df=wine, tf=False, lem=False)
    # sim_wines = top_n_sim(cvcs, 1)
    # print sim_wines
    # top_wine = pred_one(cs, 1)

    # cs.shape
    #
    # twenty5 = wine['variety'][0:25]
    # for i , doc1 in enumerate(twenty5):
    #     for j, doc2 in enumerate(twenty5):
    #         if i != j and cs[i,j] > 0.25:
    #             print i, doc1, j, doc2, cs[i, j]
    #



######################################

'''
Code for comparing descriptions if the user puts in a description

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
'''
