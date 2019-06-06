from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from task1.src.garcon import Garcon
from task1.src.learners.abstract_learner import AbstractLearner

gc = Garcon()

class DecisionTree(AbstractLearner):
	def __init__(self, X_train, y_train):
		super().__init__(X_train, y_train)
		self.model = DecisionTreeClassifier()
		self.fit()

	def set_name(self):
		self.name = 'Decision Tree'

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def classify(self, X):
		X = self.prep.processTweets(X)
		return self.model.predict(X)

	def report(self, X, y):
		gc.log()
		y_pred = self.classify(X)
		print(confusion_matrix(y, y_pred))
		print(accuracy_score(y, y_pred))