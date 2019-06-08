from task1.src.garcon import Garcon
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from task1.src.learners.abstract_learner import LearnerWrapper
from task1.src.preproccesData import TweetsPreProcessor
from sklearn.linear_model import LogisticRegression
from task1.src.dataGetter import DataGetter
from sklearn.pipeline import Pipeline
import numpy as np

gc = Garcon()


def classify(models):
    """
    Run on models from models list, fits both the pipeline on these models and LearnerWrapper
    Make prediction as the most frequent prediction of all got classifiers
    Prints classification report for train and for test and confusion matrix for test
    :param models: list of models for including to model
    :return:
    """
    gc.enter_func()
    prep = DataGetter()
    X, y = prep.get_train_data()
    X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
    # models = [LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, MultinomialNB]
    prep = TweetsPreProcessor(X_train)
    vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    ytest = np.array(y_test)
    ytrain = np.array(y_train)
    list_of_counters_train = [Counter() for i in range(X_train.shape[0])]
    list_of_counters_test = [Counter() for i in range(X_test.shape[0])]
    for i, model in enumerate(models):
        print("\n\nPerformance of " + str(model.__name__) + " model:\n\n")
        pipeline = Pipeline([('vect', vectorizer), ('clf', model())])
        # fitting pipeline model
        pipeline_model = pipeline.fit(X_train, y_train)
        for index, prediction in enumerate(pipeline_model.predict(X_train)):  # counting predictions
            list_of_counters_train[index][prediction] += 1
        for index, prediction in enumerate(pipeline_model.predict(X_test)):  # counting predictions
            list_of_counters_test[index][prediction] += 1
        # fitting our model
        learner = LearnerWrapper(model, prep, model.__name__)
        learner.fit(X_train, y_train)
        for index, prediction in enumerate(learner.predict(X_train)):  # counting predictions
            list_of_counters_train[index][prediction] += 1
        for index, prediction in enumerate(learner.predict(X_test)):  # counting predictions
            list_of_counters_test[index][prediction] += 1
    for i, counter in enumerate(list_of_counters_train):  # replacing each tweet by the most frequent prediction
        list_of_counters_train[i] = sorted(counter, key=counter.get, reverse=True)[0]
    for i, counter in enumerate(list_of_counters_test):  # replacing each tweet by the most frequent prediction
        list_of_counters_test[i] = sorted(counter, key=counter.get, reverse=True)[0]
    print(classification_report(ytrain, list_of_counters_train))
    print(classification_report(ytest, list_of_counters_test))
    print(confusion_matrix(ytest, list_of_counters_test))


classify([LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, MultinomialNB])
