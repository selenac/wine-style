from data_processing import clean_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def latentdirichletallocation(column, stop_words, num_topics=9, passes=20):
    '''
    Input:
        column: text from dataframe
        stop_words: nltk stop words plus wine specific
    Output:
        cv_docs: Matrix
        feature_names: list of features/vocab
        lda: fitted model object
    '''
    desc = get_corpus(column)
    vect, cv_docs, feature_names = get_countvect(desc, stop_words)
    lda = fit_LDA(X=cv_docs, num_topics=num_topics, passes=passes)
    return cv_docs, feature_names, lda

def display_topics(model, feature, n_words):
    for i, topic in enumerate(model.components_):
        print("Topic %d:" % (i))
        print(" ".join([feature[j]
                        for j in topic.argsort()[:-n_words - 1:-1]]))

##############################################################

def get_corpus(column):
    '''
    Input: Column of text (using: df['description'])
    Output: Corpus of text descriptions (list)
    '''
    return [desc for desc in column]

def get_countvect(desc, wine_stop_words, stemlem=0):
    '''
    Input:
        desc: tokenized descriptions
        stop_words: nltk stop words plus wine specific
        stemlem: int (0s) None, (1) Lemmatize, (2) Porter, (3) Snowball
    Output:
        vect: text CountVectorizer object
        cv_docs: sparse matrix of counts
    '''
    vect = CountVectorizer(stop_words = wine_stop_words,
                           analyzer='word',
                           decode_error = 'ignore',
                           strip_accents = 'unicode',
                           lowercase = True)
                           # max_df = 0.97, # min_df = 0.03, # ngram_range = (1,2)
    cv_docs = vect.fit_transform(desc)
    vocab = vect.get_feature_names()
    return vect, cv_docs, vocab

def fit_LDA(X, num_topics=9, passes=20):
    '''
    Input:
        X: vectorized matrix of the documents
        num_topics: (int) number of topic categories
        passes: (int) number of iterations to fit
    Output: lda object model
    '''
    print 'fitting...'
    lda = LatentDirichletAllocation(n_components=num_topics,
                                    max_iter=passes,
                                    learning_method='online').fit(X)
    return lda

# Gensim LDA model not working [issue: vocab not relevant?]

# from gensim import matutils
# from gensim.models.ldamodel import LdaModel
# def fit_LDA(X, vocab, num_topics=10, passes=10):
# return LdaModel(matutils.Sparse2Corpus(X),
#                 num_topics=num_topics, passes=passes,
#                 id2word=dict([(i, s) for i, s in enumerate(vocab)]))

# def print_topics(lda, vocab, n=10):
#     topics = lda.show_topics()
#      gensim.matutils.argsort(x, topn=None, reverse=False)

if __name__ == '__main__':
    filepath = '../../data/sample.csv'
    wine_df, wine_stop_lib = clean_data(filepath)

    documents = get_corpus(wine_df['description'])
    vect, cv_docs, vocab = get_countvect(documents, wine_stop_lib)

    lda = fit_LDA(X=cv_docs, num_topics=9, passes=25)
    display_topics(model=lda, feature=vocab, n_words=10)
    lda_map = lda.fit_transform(cv_docs) #returns numpy array map of (wine x topics)

    '''
    lda_map[0].argmax() #return top topic index col
    lda_map[0].max() # return top topic score
    lda_map[0].argsort()[-3:][::-1] #return top 3 topics index col
    '''
