from abc import ABC, abstractmethod

class AbstractLearner(ABC):
	@abstractmethod
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	@abstractmethod
	def predict(self, X):
		pass

	@abstractmethod
	def report(self):
		pass
