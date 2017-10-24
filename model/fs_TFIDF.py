import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_matrix_features(column, stop_words, stemlem=0):
    '''
    Input:
        column: text from dataframe
        stop_words: nltk stop words plus wine specific
        stemlem: int (0s) None, (1) Lemmatize, (2) Porter, (3) Snowball
    Output:
        tfidf_docs: Matrix
        feature_names: array of features/vocab
    '''
    desc = get_corpus(column)
    vect, tfidf_docs = get_tfidf(desc, stop_words, stemlem)
    feature_names = np.array(vect.get_feature_names())
    return tfidf_docs, feature_names

# TODO top features not working

def find_top_features_per_wine(feature_names, tfidf_docs, n_features=10):
    '''
    Input:
        feature_names, tfidf_docs, and n_features (number of features)
    Output:
        top n features from each wine in TFIDF matrix, and wine index
    '''
    tfidf_docs = tfidf_docs.toarray()
    top_idx = np.empty([tfidf_docs.shape[0], n_features], dtype=int)
    top_feats = np.empty([tfidf_docs.shape[0], n_features], dtype=object)
    for i, row in enumerate(tfidf_docs):
        top_idx[i] = np.argsort(row)[-n_features:][::-1]
        top_feats[i] = feature_names[top_idx[i]]
    return top_feats, top_idx

def find_top_features_all_wines(vectorizer, n=5):
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    return [features[i] for i in indices[:n]]

###########################################################################

def get_corpus(column):
    '''
    Input: Column of text (using: df['description'])
    Output: Corpus of text descriptions (list)
    '''
    return [desc for desc in column]

''' Various Tokenizers '''
def regular_tokens(descriptions):
    pass

def _lemmatize_tokens(descriptions):
    l = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(l.lemmatize(w) for w in d.split())
    return [lemmatize(desc.decode(errors='ignore')) for desc in descriptions]

def _porter_stem_tokens(descriptions):
    p = PorterStemmer()
    porter = lambda d: " ".join(p.stem(w) for w in d.split())
    return [porter(desc.decode(errors='ignore')) for desc in descriptions]

def _snowball_stem_tokens(descriptions):
    s = SnowballStemmer('english')
    stemmer = lambda d: " ".join(s.stem(w) for w in d.split())
    return [stemmer(desc.decode(errors='ignore')) for desc in descriptions]

''' Vectorizer '''

def get_tfidf(desc, wine_stop_words, stemlem=0, max_features=1000):
    '''
    Input:
        desc: tokenized descriptions
        stop_words: nltk stop words plus wine specific
        stemlem: int (0s) None, (1) Lemmatize, (2) Porter, (3) Snowball
    Output:
        vect: tfidf text Vectorizer object
        tfidf_docs: sparse matrix of TFIDF values

    '''
    vect = TfidfVectorizer(stop_words = wine_stop_words,
                            decode_error = 'ignore',
                            strip_accents = 'unicode',
                            max_df = 0.99,
                            min_df = 0.01,
                            ngram_range = (1,2),
                            lowercase = True)
    if stemlem==1:
        desc = _lemmatize_tokens(desc)
    elif stemlem==2:
        desc = _porter_stem_tokens(desc)
    elif stemlem==3:
        desc = _snowball_stem_tokens(desc)
    tfidf_docs = vect.fit_transform(desc)
    return vect, tfidf_docs

if __name__ == '__main__':
    pass
