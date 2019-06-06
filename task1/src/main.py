from baseline import *
from preprocessor import *
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = 'tweets_data'
gc  = Garcon()

def classifier_report(classifier, X_test, y_test):
	'''
	Creates reports for a given classifier.
	:param classifier:		Classifier to report.
	:param X_test: 			shape=(n_samples, )
	:param y_test: 			shape=(n_samples, )
	'''
	gc.enter_func()
	y_hat = classifier.classify(X_test)
	print(classification_report(y_test, y_hat))
	print(confusion_matrix(y_test, y_hat))


def main():
	gc.enter_func()
	prep = Preprocessor()
	X, y = prep.get_train_data()

	X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
	baseliner = Baseline(X_train, y_train)
	baseliner.report(X_test, y_test)


if __name__ == '__main__':
	main()
