import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

'''
Not using, yet!

import numpy as np
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
'''
# class TransformText(object):
#     def __init__(self):
#         pass
#
#     def fit(self, X, y):
#         pass
#
#     def transform(self, X, y):
#         pass


def load_data(filename):
    '''
    input: file csv format
    output: descriptions as features column, and targets as label column
    '''
    df = pd.read_csv(filename)
    descriptions = df['description']
    targets = df['variety']
    return descriptions, targets

# def tokenize(column):
#     tokenizer = RegexpTokenizer('\w+\S')
#     tokenized = [tokenizer.tokenize(column[i].lower()) for i in xrange(len(column))]
#     stop = set(stopwords.words('english') + list(string.punctuation))
#     tokenized = [[word for word in words if word not in stop]for words in tokenized]
#     return tokenized

# def lem_words(descriptions):
#     '''
#     input: description list
#     output: list of lem words
#     '''
#     wordnet = WordNetLemmatizer()
#     lem_tokenized = [[wordnet.lemmatize(word) for word in words] for words in tokenized]
#     return lem_tokenized

def get_tfidf_vect(descriptions, num_features=100):
    '''
    input: raw document - descriptions (training or test x variables)
    output: sparse matrix
    '''
    vect = TfidfVectorizer(max_features=num_features,
                           stop_words='english',
                           decode_error='ignore',
                           strip_accents='unicode',
                           lowercase=True)
    tfidf = vect.fit_transform(descriptions)
    return tfidf

def get_count_vect(descriptions, num_features=100):
    '''
    input: raw document - descriptions (training or test x variables)
    output: sparse matrix
    '''
    vect = CountVectorizer(max_features=num_features,
                           stop_words='english',
                           decode_error='ignore')
    return vect.fit_transform(descriptions)

def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_test, y_predict, average='weighted'), \
        precision_score(y_test, y_predict, average='weighted'), \
        recall_score(y_test, y_predict, average='weighted')

def run_test(models, desc_train, desc_test, y_train, y_test):
    X_train = get_tfidf_vect(desc_train)
    X_test = get_tfidf_vect(desc_test)
    #X_train = get_count_vect(desc_train)
    #X_test = get_count_vect(desc_test)

    print "acc\tf1\tprec\trecall"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


if __name__ == '__main__':
    # models to run
    models = [LogisticRegression, MultinomialNB]
    #KNeighborsClassifier, MultinomialNB,  RandomForestClassifier]

    # data file for training
    filename = 'eda/train_sample.csv'
    # extract review text and varietal labels from data set
    descriptions, labels = load_data(filename)

    # split descriptions and labels in training and test/validation set
    desc_train, desc_test, y_train, y_test = train_test_split(descriptions, labels)
    # run the models and report scores
    run_test(models, desc_train, desc_test, y_train, y_test)


#
#
# from sklearn.metrics.pairwise import linear_kernel
#
# tfidf = get_tfidf_vect(desc_train)
# cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
# related_docs_indices = cosine_similarities.argsort()[:-5:-1]
# related_docs_indices
# cosine_similarities[related_docs_indices]
#
# desc_train[related_docs_indices]
# y_train[related_docs_indices]
#
#
# '''
#     # Transform
#     X_train = get_tfidf_vect(desc_train)
#     X_test = get_tfidf_vect(desc_test)
# '''
#
# '''
#     # Classifier
#     log = LogisticRegression()
#     log.fit(X_train, y_train)
#     pred_x = log.predict(X_test)
#     log.score(X_test, y_test) # 0.2115
# '''
