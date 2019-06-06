from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from garcon import Garcon


gc = Garcon()

class AdaBoost():
	def __init__(self, learner, X_train, y_train, n_estimators=50,
				 learning_rate=1):
		gc.enter_func()
		self.T = n_estimators
		self.lr = learning_rate
		self.learner = learner
		self.ada = AdaBoostClassifier(self.learner, n_estimators, learning_rate)
		self.model = self.ada.fit(X_train, y_train)

	def classify(self, X):
		gc.enter_func()
		y_hat = self.model.predict(X)
		return y_hat

	def confusion_matrix(self, X_test, y_test):
		gc.enter_func()
		y_pred = self.classify(X_test)
		return metrics.confusion_matrix(y_test, y_pred)

	def classification_report(self, y_test, X_test):
		gc.enter_func()
		y_pred = self.classify(X_test)
		return metrics.classification_report(y_test, y_pred)