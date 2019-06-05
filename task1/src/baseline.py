from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from garcon import Garcon

gc = Garcon()

class Baseline:
	def __init__(self, X_train, y_train):
		self.X_train = self.preprocess(X_train)
		self.y_train = y_train
		self.model = LinearRegression()
		self.train()

	def preprocess(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return: 	shape=(n_samples, n_features)
		'''
		gc.enter_func()
		vectorizer = CountVectorizer()
		X_vecd = vectorizer.fit_transform(X)
		return X_vecd.toarray()

	def train(self):
		gc.enter_func()
		self.model.fit(self.X_train, self.y_train)

	def classify(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return:	shape=(n_samples, )
		'''
		gc.enter_func()
		X = self.preprocess(X)
		return self.model.predict(X)
