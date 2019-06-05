from baseline import *
from preprocessor import *
from sklearn.metrics import confusion_matrix, classification_report

def classifier_report(classifier, X_test, y_test):
	y_hat = classifier.classify(X_test)
	print(classification_report(y_test, y_hat))
	print(confusion_matrix(y_test, y_hat))


def main(X, y):
	prep = Preprocessor()
	X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
	baseliner = Baseline(X_train, y_train)
	classifier_report(baseliner, X_test, y_test)

if __name__ == '__main__':
	main()
