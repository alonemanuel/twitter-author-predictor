import string

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,\
	HashingVectorizer
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from src.garcon import Garcon

gc = Garcon()

class Baseline:
	pat1 = r'@[A-za-z0-9]+'
	pat2 = r'https?://[A-Za-z0-9./]+'
	pat3 = r'^https?:\/\/.*[\r\n]*'
	pat_num = r'\b\d+'
	pat_ing = r'ing\b'
	pat_test_a = r'[\w]*a[\w]*'
	pat_test_e = r'[\w]*e[\w]*'
	pat_test_o = r'[\w]*o[\w]*'
	combined_pat = r'|'.join((pat1, pat2, pat3, pat_num, pat_ing))

	def __init__(self, X_train, y_train):
		'''
		:param X_train:		shape=(n_samples, )
		:param y_train: 	shape=(n_samples, )
		'''
		gc.enter_func()
		# self.X_train = self.preprocess(X_train)
		self.X_train = X_train
		self.y_train = y_train.astype(int)
		self.model = svm.SVC(gamma='scale', decision_function_shape='ovo')
		self.train()
		gc.log('Done train')

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

		# vectorizer = TfidfVectorizer()
		vectorizer = CountVectorizer()
		# vectorizer = HashingVectorizer(n_features=20)
		# X_vecd = vectorizer.transform(X)
		X_vecd = vectorizer.fit_transform(X)
		print(X_vecd.shape)
		print(X_vecd)
		# print(vectorizer.get_feature_names())
		return X_vecd.toarray()

	def train(self):
		gc.enter_func()
		print(self.y_train)
		self.model.fit(self.X_train, self.y_train)
		gc.log('Done training')
		# plt.boxplot(self.y_train)
		# plt.show()

	# self.model.fit(self.X_train, self.y_train)

	def classify(self, X):
		'''
		:param X:	shape=(n_samples, )
		:return:	shape=(n_samples, )
		'''
		gc.enter_func()
		X = self.preprocess(X)
		return self.model.predict(X)
