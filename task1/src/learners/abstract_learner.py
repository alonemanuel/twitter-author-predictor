from abc import ABC, abstractmethod
from preprocessor import Preprocessor

class AbstractLearner(ABC):
	@abstractmethod
	def __init__(self, X_train, y_train):
		self.prep = Preprocessor()
		self.X_train = self.prep.prep_tweets(X_train)
		self.y_train = y_train

	@abstractmethod
	def classify(self, X):
		pass

	@abstractmethod
	def report(self, X, y):
		pass
