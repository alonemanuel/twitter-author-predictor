from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

class LogisticReg:
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		self.classifier = ClassifierChain(LogisticRegression())
		self.fit()
		pass

	def fit(self):
		self.classifier.fit(self.X_train, self.y_train)

	def classify(self, X):
		y_pred = self.classifier.predict(X)
		return y_pred

