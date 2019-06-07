from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from task1.src import learner_wrapper
from task1.src.garcon import Garcon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from task1.src.data_getter import DataGetter

from sklearn.pipeline import Pipeline
import pickle

from task1.src.learner_wrapper import LearnerWrapper
from task1.src.preprocessor import Preprocessor

gc = Garcon()


def main():
	'''
	Runs everything, and in particular creates a class of hypotheses and
	creates a report for each.
	'''
	gc.enter_func()

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
						 ('chi', SelectKBest(chi2, k=17000)),
						 ('clf', LogisticRegression(solver='sag', penalty='l2',
													tol=1e-5))])
	# fitting our model and save it in a pickle for later use
	classifier = pipeline.fit(X_train, y_train)
	learner_wrapper.report_predictor(classifier, 'Pipeline', X_train, y_train)
	learner_wrapper.report_predictor(classifier, 'Pipeline', X_test, y_test, is_test=True)
	# classifier_f = open('pipeline_logreg.pickle', 'rb')
	# classifier = pickle.load(classifier_f)
	# classifier_f.close()
	ytest = np.array(y_test)

	# pipe_wrap = LearnerWrapper(pipeline, prep, 'Pipeline')
	# pipe_wrap.report(X_train, y_train)
	# pipe_wrap.report(X_test, y_test, is_test=True)

	# confusion matrix and classification report(precision, recall, F1-score)
	# print(classification_report(y_train, classifier.predict(X_train)))
	# print(classification_report(ytest, classifier.predict(X_test)))
	# print(confusion_matrix(ytest, classifier.predict(X_test)))

	# models = {'Logistic Regression': LogisticRegression, 'K-Nearest Neighbors': KNeighborsClassifier,
	# 		  'Decision Trees Classifier': DecisionTreeClassifier}
	models = {'Logistic Regression': LogisticRegression, 'Multinomial Naive Bayes':MultinomialNB}
	learners = [None] * len(models)

	i = 0
	for name, model in models.items():
		learners[i] = LearnerWrapper(model, prep, name)
		i += 1

	for learner in learners:
		learner.fit(X_train, y_train)
		learner.report(X_train, y_train)
		learner.report(X_test, y_test, is_test=True)


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
