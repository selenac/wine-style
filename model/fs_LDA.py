from data_processing import clean_data

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

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
                           #decode_error = 'ignore',
                           #strip_accents = 'unicode',
                           # max_df = 0.97, # min_df = 0.03, # ngram_range = (1,2),
                           lowercase = True)
    cv_docs = vect.fit_transform(desc)
    vocab = vect.get_feature_names()
    return vect, cv_docs, vocab

def fit_LDA(X, vocab, num_topics=10, passes=10):
    '''
    '''
    print 'fitting lda...'
    return LdaModel(matutils.Sparse2Corpus(X),
                    num_topics=num_topics, passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))

# def print_topics(lda, vocab, n=10):
#     '''
#     '''
#     topics = lda.show_topics()
# gensim.matutils.argsort(x, topn=None, reverse=False)

if __name__ == '__main__':
    filepath = '../../data/sample.csv'
    wine_df, wine_stop_lib = clean_data(filepath)

    documents = get_corpus(wine_df['description'])
    vect, cv_docs, vocab = get_countvect(documents, wine_stop_lib)

    lda = fit_LDA(cv_docs, vocab, num_topics=5, passes=20)
    lda.print_topics()
