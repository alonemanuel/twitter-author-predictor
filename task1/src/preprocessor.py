from sklearn.model_selection import train_test_split
import numpy as np

TRAIN_RATIO = 0.85

class Preprocessor:

	def train_test_split(self, X, y):
		'''
		:param X:	shape=(n_samples, )
		:param y:	shape=(n_samples, )
		:return:	X_train, y_train, X_test, y_test, all with
					shape=(n_samples, )
		'''
		stacked = np.stack((X, y), axis=1)
		train, test = train_test_split(stacked, train_size=(TRAIN_RATIO),
									   shuffle=True)
		X_train, y_train = train[0], train[1]
		X_test, y_test = test[0], test[1]
		return X_train, y_train, X_test, y_test
