# import os
#
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# from garcon import Garcon
#
# gc = Garcon()
# CSV_EXT = ".csv"
#
# TRAIN_RATIO = 0.85
# DATA_DIR_PATH = 'tweets_data'
#
# class Preprocessor:
#
# 	def get_train_data(self):
# 		'''
# 		:return:	data. shape=(n_samples, 2)
# 		'''
# 		gc.enter_func()
# 		dfs = []
# 		for fn in os.listdir(DATA_DIR_PATH):
# 			if fn.endswith(CSV_EXT) and not fn.startswith('tweets_test'):
# 				dfs.append(pd.read_csv(os.path.join(DATA_DIR_PATH, fn)))
# 		df = pd.concat(dfs)
# 		return df.values[:, 1], df.values[:, 0]
#
# 	def train_test_split(self, X, y):
# 		'''
# 		:param X:	shape=(n_samples, )
# 		:param y:	shape=(n_samples, )
# 		:return:	X_train, y_train, X_test, y_test, all with
# 					shape=(n_samples, )
# 		'''
# 		gc.enter_func()
# 		stacked = np.stack((X, y), axis=1)
# 		train, test = train_test_split(stacked, train_size=TRAIN_RATIO,
# 									   test_size=1 - TRAIN_RATIO, shuffle=True)
# 		X_train, y_train = train[:, 0], train[:, 1]
# 		X_test, y_test = test[:, 0], test[:, 1]
# 		return X_train, y_train, X_test, y_test



import os

from sklearn.model_selection import train_test_split
import pandas as pd

CSV_EXT = ".csv"

TRAIN_RATIO = 0.85
DATA_DIR_PATH = 'tweets_data'

class DataGetter:

	def get_train_data(self):
		'''
		:return:	data. shape=(n_samples, 2)
		'''
		dfs = []
		for fn in os.listdir(DATA_DIR_PATH):
			if fn.endswith(CSV_EXT) and not fn.startswith('tweets_test'):
				dfs.append(pd.read_csv(os.path.join(DATA_DIR_PATH, fn)))
		df = pd.concat(dfs)
		X = df['tweet']
		y = df['user']

		return X, y

	def train_test_split(self, X, y):
		'''
		:param X:	shape=(n_samples, )
		:param y:	shape=(n_samples, )
		:return:	X_train, y_train, X_test, y_test, all with
					shape=(n_samples, )
		'''
		X_train, X_test, y_train, y_test = train_test_split(X, y,
															train_size=TRAIN_RATIO,
															test_size=1 - TRAIN_RATIO,
															shuffle=True)
		return X_train, y_train, X_test, y_test