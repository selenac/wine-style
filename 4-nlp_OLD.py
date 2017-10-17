import nltk
import numpy as np
import pandas as pd
import string

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import MultinomialNB

def load_data(filename):
    '''
    input: path to the file
    output: dataframe

    remove duplicates from data
    '''
    df = pd.read_csv(filename).drop_duplicates().reset_index()
#   df = df.drop('index', axis=1, inplace=True)
    #, encoding='utf-8')
    return df

def make_dict(df, column, n=10000):
    '''
    input:
        df - dataframe
        column - column name of text evaluating
        n  - integer for number of words in dictionary
    output:
        dictionary of top n words
        updated dataframe with tokenized column
    '''
    word_list = tokenize(df[column])
    # lem_word_list = lem_words(word_list)
    df = tokenize_column(df, word_list)
    word_dict = wine_dictionary(word_list, n)
    return word_dict, df

def features(desc, word_dict):
    '''
    input: tokenized document
    output: features dictionary
    '''
    words = set(desc)
    features = {}
    for w in word_dict:
        features[w] = bool(w in words)
    return features

featureset = [(features(wine_samp['tokenized'][i], samp_dict), wine_samp['variety'][i])
            for i in xrange(len(wine_samp))]

classifier = NaiveBayesClassifier.train(labeled_featuresets=featureset)

classifier.show_most_informative_features(50)

def old_tokenize(column):
    '''
    input: dataframe
    output: list words from descriptions (remove stop words)

    break out words for each row in descriptions column
    '''
    tokenized = [column[i].lower().split() for i in xrange(len(column))]
    stop = set(stopwords.words('english') + list(string.punctuation))
    tokenized = [[word for word in words if word not in stop]for words in tokenized]
    return tokenized

def tokenize(column):
    tokenizer = RegexpTokenizer('\w+\S')
    tokenized = [tokenizer.tokenize(column[i].lower()) for i in xrange(len(column))]
    stop = set(stopwords.words('english') + list(string.punctuation))
    tokenized = [[word for word in words if word not in stop]for words in tokenized]
    return tokenized

def tokenize_column(df, tokenized):
    '''
    input: dataframe and tokenized descriptions
    output: append tokenized column to dataframe
    '''
    df['tokenized'] = tokenized
    return df

def lem_words(tokenized):
    '''
    input: description list
    output: list of lem words
    '''
    wordnet = WordNetLemmatizer()
    lem_tokenized = [[wordnet.lemmatize(word) for word in words] for words in tokenized]
    return lem_tokenized

def wine_dictionary(tokenized, n):
    '''
    input: dataframe, n (number of most frequent words)
    output: dictionary with n most frequent words and counts
    '''
    words = []
    for desc in tokenized:
        for word in desc:
            words.append(word)
    words = FreqDist(words)
    top_words = dict(words.most_common(n))
    return top_words


def vocab(lem_descriptions):
    '''
    input: list of lem words
    output: list of unique words
    '''
    vocab_set = set()
    [[vocab_set.add(token) for token in tokens] for tokens in lem_descriptions]
    return list(vocab_set)

def word_matrix(vocab_set):
    matrix = [[0] * len(vocab_set) for doc in descriptions]
    vocab_dict = dict((word, i) for i, word in enumerate(vocab_set))
    for i, words in enumerate(lem_descriptions):
        for word in words:
            matrix[i][vocab_dict[word]] += 1

def categories(df, n=40):
    '''
    input: DataFrame, n integer for # of varietals
    output: categories dataframe with count of wine in that category.
    '''
    varieties = df[['variety', 'country']].groupby('variety').count()
    categories = varieties[(varieties['country'] > n)]
    return categories






cv = CountVectorizer(stop_words='english', decode_error='ignore')
vectorized = cv.fit_transform(wine_samp['description'])

vectorized.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(vectorized)
X_train_tfidf.shape

y_targ = wine_samp['variety']

clf = MultinomialNB().fit(X_train_tfidf, y_targ)

wine_test2 = load_data('eda/samp_test.csv')
x_test = cv.fit_transform(wine_test2['description'])
x_test_tfidf = tfidf_transformer.fit_transform(x_test)
predict = clf.predict(x_test_tfidf)


tfidf = TfidfVectorizer(stop_words='english')
tfidfed = tfidf.fit_transform(wine_test['description'])

y_train = np.array(wine_test['variety'])
X_train = np.array(wine_test['tokenized'])

mnb = MultinomialNB()
mnb.fit(X_train, y_train)



X_train
mnb.score(X_test, y_test)
sklearn_predictions = mnb.predict(X_test)

cosine_similarities = linear_kernel(tfidfed, tfidfed)




if __name__ == '__main__':
    # filename = 'data/wine_data_2017.csv'
    # wine_df = load_data(filename)
    # wine_dict, wine_df = make_dict(wine_df, 'description')
    # wine_dict


    wine_samp = load_data('eda/samp_train.csv')
    wine_test = load_data('eda/samp_test.csv')

    samp_dict, wine_samp = make_dict(wine_samp, 'description')
    samp_dict

    len(samp_dict)
    wine_samp['tokenized'][0]
    def doc_feats(doc, w_d):
        doc_words = set(doc)
        features = {}
        for word in w_d:
            features[word] = (word in doc_words)
        return features


    a =  doc_feats(wine_samp['tokenized'], samp_dict)
    a['balanced']


    # tokenized = tokenize(wine_df['description'])
    # wine_df = tokenize_column(wine_df, tokenized)
#     word_tokenize(wine_df['description'][0].lower())
#     wine_df.apply(lambda text: word_tokenize(unicode(text, 'utf-8')))
#     docs = [word_tokenize(content) for content in wine_df['description']]
#     wine_df['tokenized'] = wine_df['description'].apply(word_tokenize)
