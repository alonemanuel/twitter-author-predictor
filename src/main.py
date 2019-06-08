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
import pickle

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

	vectorizer = TfidfVectorizer(min_df=3, stop_words="english",
								 sublinear_tf=True, norm='l2',
								 ngram_range=(1, 2))

	# vectorizer

	pipeline = Pipeline([('vect', vectorizer),
						 ('chi', SelectKBest(chi2, k=18000)),
						 ('clf', LogisticRegression(solver='sag', penalty='l2',
													tol=1e-5))])
	# fitting our model and save it in a pickle for later use
	# classifier = pipeline.fit(X_train, y_train)
	classifier_f = open('pipeline_logreg.pickle', 'rb')
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	ytest = np.array(y_test)

	# confusion matrix and classification report(precision, recall, F1-score)
	print(classification_report(y_train, classifier.predict(X_train)))
	print(classification_report(ytest, classifier.predict(X_test)))
	print(confusion_matrix(ytest, classifier.predict(X_test)))

	# save_classifier = open('pipeline_logreg.pickle','wb')
	# pickle.dump(classifier, save_classifier)
	# save_classifier.close()


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
