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
import numpy as np
import pandas as pd
from garcon import Garcon

gc = Garcon()
CSV_EXT = ".csv"

TRAIN_RATIO = 0.85
DATA_DIR_PATH = 'tweets_data'

class Preprocessor:

	def get_train_data(self):
		'''
		:return:	data. shape=(n_samples, 2)
		'''
		gc.enter_func()
		dfs = []
		for fn in os.listdir(DATA_DIR_PATH):
			if fn.endswith(CSV_EXT) and not fn.startswith('tweets_test'):
				dfs.append(pd.read_csv(os.path.join(DATA_DIR_PATH, fn)))
		df = pd.concat(dfs)
		y = df['user']
		X = df['tweet']

		return X, y

	def train_test_split(self, X, y):
		'''
		:param X:	shape=(n_samples, )
		:param y:	shape=(n_samples, )
		:return:	X_train, y_train, X_test, y_test, all with
					shape=(n_samples, )
		'''
		gc.enter_func()
		X_train, X_test, y_train, y_test = train_test_split(X, y,
															train_size=TRAIN_RATIO,
															test_size=1 - TRAIN_RATIO,
															shuffle=True)
		return X_train, y_train, X_test, y_test

	def get_char_count(self, tweet):
		'''
		Returns the char count of the tweet (of all chars).
		:return:
		'''
		return len(tweet)

	def get_word_count(self, tweet):
		'''
		Return word count for tweet.
		:param tweet:
		:return:
		'''
		return len(tweet.split())

	def get_hash_count(self, tweet):
		'''
		Return # count for tweet.
		:param tweet:
		:return:
		'''
		return tweet.count('#')

	def get_at_count(self, tweet):
		'''
		Return @ count for tweet.
		:param tweet:
		:return:
		'''
		return tweet.count('@')

	def prep_tweets(self, tweets):
		'''
		Preprocesses tweet(s).
		:param X: 	shape=(n_samples, )
		:return: 	shape=(n_samples, n_features)
		'''
		char_counts = []
		word_counts = []
		hash_counts = []
		at_counts = []
		emoji_count = []

		for tweet in tweets:
			# tweet = row['tweet']
			char_counts.append(self.get_char_count(tweet))
			word_counts.append(self.get_word_count(tweet))
			hash_counts.append(self.get_hash_count(tweet))
			at_counts.append(self.get_at_count(tweet))
		df = pd.DataFrame()
		df['char_count'] = char_counts
		df['word_count'] = word_counts
		df['hash_count'] = hash_counts
		df['at_count'] = at_counts
		return df.values