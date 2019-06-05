from sklearn.feature_extraction.text import  CountVectorizer

class baseline:
	def __init__(self, X_train, y_train):
		self.X_train = self.preprocess(X_train)
		self.y_train = y_train
		self.train()

	def preprocess(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return: 	shape=(n_samples, n_features)
		'''
		vectorizer = CountVectorizer()
		X_vecd = vectorizer.fit_transform(X)
		return X_vecd.toarray()

	def train(self):

		pass

	def predict(self):
		pass