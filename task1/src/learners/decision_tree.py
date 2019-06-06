from sklearn.tree import DecisionTreeClassifier

from learners.abstract_learner import AbstractLearner

class DecisionTree(AbstractLearner):
	def __init__(self, X_train, y_train):
		super().__init__(X_train, y_train)
		self.model = DecisionTreeClassifier()
		self.fit()

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def predict(self, X):
		return self.model.predict(X)

	def report(self):
		pass


