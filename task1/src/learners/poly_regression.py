import numpy as np
from sklearn import metrics, datasets, linear_model
from sklearn.model_selection import cross_validate
from garcon import Garcon

gc = Garcon()

class PolyRegression:
	DEG = 3

	def __init__(self, X_train, y_train):
		'''
		:param X_train: shape=(n_samples, n_features)
		:param y_train: shape=(n_samples, )
		'''
		gc.enter_func()
		self.X_train = X_train
		self.y_train = y_train
		self.poly = np.polyfit(X_train, y_train, deg=self.DEG)
		self.fit()

	def fit(self):
		pass

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

	def report(self, X):
		pass