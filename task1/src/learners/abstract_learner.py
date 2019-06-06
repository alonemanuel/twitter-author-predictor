from abc import ABC, abstractmethod

from garcon import Garcon
from preprocessor import Preprocessor
import matplotlib.pyplot as plt

gc = Garcon()

class AbstractLearner(ABC):
	@abstractmethod
	def __init__(self, X_train, y_train):
		self.prep = Preprocessor()
		self.X_train = self.prep.prep_tweets(X_train)
		self.y_train = y_train
		self.classifier_name=''

	@abstractmethod
	def set_name(self):
		pass

	@abstractmethod
	def classify(self, X):
		pass

	@abstractmethod
	def report(self, X, y):
		pass
