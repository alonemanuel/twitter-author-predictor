import string

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from garcon import Garcon

gc = Garcon()

class Baseline:
	pat1 = r'@[A-za-z0-9]+'
	pat2 = r'https?://[A-Za-z0-9./]+'
	pat3 = r'^https?:\/\/.*[\r\n]*'
	pat_num = r'\b\d+'
	pat_ing = r'ing\b'
	combined_pat = r'|'.join((pat1, pat2, pat3, pat_num, pat_ing))

	def __init__(self, X_train, y_train):
		'''
		:param X_train:		shape=(n_samples, )
		:param y_train: 	shape=(n_samples, )
		'''
		gc.enter_func()
		self.X_train = self.preprocess(X_train)
		gc.log(self.X_train.shape)
		self.y_train = y_train
		self.model = svm.SVC(gamma='scale', decision_function_shape='ovo')
		self.train()

	def clean(self, str):
		'''
		:param str:	string
		:return:	cleaned string
		'''
		stripped = re.sub(self.combined_pat, '', str, flags=re.MULTILINE)
		lowered= stripped.lower()
		return lowered

	def preprocess(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return: 	shape=(n_samples, n_features)
		'''
		gc.enter_func()
		gc.iter(X)
		for i, tweet in enumerate(np.ndarray.tolist(X)):
			X[i] = self.clean(tweet)
			gc.iter()

		vectorizer = CountVectorizer()
		X_vecd = vectorizer.fit_transform(X)
		print(vectorizer.get_feature_names())
		return X_vecd.toarray()

	def train(self):
		gc.enter_func()
		plt.boxplot(self.y_train)
		plt.show()

	# self.model.fit(self.X_train, self.y_train)

	def classify(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return:	shape=(n_samples, )
		'''
		gc.enter_func()
		X = self.preprocess(X)
		return self.model.predict(X)
