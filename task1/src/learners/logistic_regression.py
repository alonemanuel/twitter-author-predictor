from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multioutput import ClassifierChain

from task1.src.garcon import Garcon
from task1.src.learners.abstract_learner import LearnerWrapper

gc = Garcon()

class LogisticReg(LearnerWrapper):
	def __init__(self, X_train, y_train):
		super().__init__(X_train, y_train)
		self.model = LogisticRegression()
		self.fit()

	def set_name(self):
		self.classifier_name = 'Logistic Regression'

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def classify(self, X):
		X = self.prep.processTweets(X)
		y_pred = self.model.predict(X)
		return y_pred

	def report(self, X, y):
		gc.log()
		y_pred = self.classify(X)
		print(confusion_matrix(y, y_pred))
		print(accuracy_score(y, y_pred))
