import pickle

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from task1.src.data_getter import DataGetter

from sklearn.pipeline import Pipeline

from task1.src.preprocessor import Preprocessor


def train_and_save_model():
    '''
    Imports data , fits our model, and saves it for later use
	'''

    data_prep = DataGetter()
    X, y = data_prep.get_train_data()
    X_train, y_train, X_test, y_test = data_prep.train_test_split(X, y)
    prep = Preprocessor(X_train, y_train)
    prep.preprocess()
    vectorizer = TfidfVectorizer(min_df=3, stop_words="english",
                                 sublinear_tf=True, norm='l2',
                                 ngram_range=(1, 2))

    # vectorizer

    pipeline = Pipeline([('vect', vectorizer),
                         ('chi', SelectKBest(chi2, k=18000)),
                         ('clf', LinearSVC(penalty='l2',
                                           tol=1e-5))])

    classifier = pipeline.fit(X_train, y_train)
    #

    save_classifier = open('classify-final-build.pickle','wb')
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

