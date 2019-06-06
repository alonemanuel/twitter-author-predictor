from task1.src.garcon import Garcon
from task1.src.learners import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import re
from sklearn.ensemble import RandomForestClassifier

from task1.src.preproccesData import TweetsPreProcessor
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

from task1.src.learners.learner_analyzer import LearnerAnalyzer
from task1.src.dataGetter import DataGetter

from sklearn.pipeline import Pipeline

gc = Garcon()


def main():
    '''
	Runs everything, and in particular creates a class of hypotheses and
	creates a report for each.
	'''
    gc.enter_func()

    prep = DataGetter()
    X, y = prep.get_train_data()
    X_train, y_train, X_test, y_test = prep.train_test_split(X, y)

    # prep = TweetsPreProcessor(X_train)
    # data = prep.processTweets(X_train, y_train)

    vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

    # vectorizer

    pipeline = Pipeline([('vect', vectorizer),
                         ('chi', SelectKBest(chi2, k=10000)),
                         ('clf', RandomForestClassifier(max_depth=500, n_estimators=15))])
                         # ('clf', LogisticRegression(solver='sag', penalty='l2', tol=1e-5))])
    # fitting our model and save it in a pickle for later use
    model = pipeline.fit(X_train, y_train)
    ytest = np.array(y_test)
    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(y_train, model.predict(X_train)))
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))


# KNN, Multinomial Naive Bayes, Linear SVC, and Random Forrest

# Helps in preprocessing tasks
# prep = Preprocessor()
# X, y = prep.get_train_data()
# X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
# # Different learning models
# models = [LogisticReg]#, RandomForest, DecisionTree, LDA]#KNN
# learners = [None, None, None]
# for i, model in enumerate(models):
# 	print("\n\nPerformance of " + str(model.__name__) + " model:\n\n")
# 	analyzer = LearnerAnalyzer(model, X_train, y_train)
# 	analyzer.report()


if __name__ == '__main__':
    main()
