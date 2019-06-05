class classifier:
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

		self.train()

	def train(self):
		pass

	def classify(self, X):
		'''
		:param X:	samples to tag. shape=(n_samples, )
		:return:	y_hat = labels (0-9) of samples. shape=(n_samples, )
		'''
		pass
