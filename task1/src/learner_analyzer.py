from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
from task1.src.garcon import Garcon
import numpy as np
from sklearn.model_selection import train_test_split
from task1.src.preprocess_data import DataPreProcessor

gc = Garcon()
TRAIN_RATIO = 0.85
STEP_SIZE=5000

class LearnerAnalyzer:
	def __init__(self, learner, X_train, y_train):
		self.learner = learner
		self.X_train = X_train
		self.y_train = y_train

	def report(self):
		self.report_bias_variance(type(self.learner).__name__)


	def report_bias_variance(self,learner_name):
		gc.enter_func()
		gc.init_plt()
		max_train_size = self.X_train.shape[0]
		# size_range = np.arange(5000, max_train_size, STEP_SIZE)
		size_range = np.arange(max_train_size, max_train_size+1)
		train_accuracy = np.zeros(size_range.shape)
		test_accuracy = np.zeros(size_range.shape)
		for i, train_size in enumerate(size_range):
			new_X = self.X_train[:train_size]
			new_y = self.y_train[:train_size]
			X_train, X_test, y_train, y_test = train_test_split(new_X, new_y,
																train_size=TRAIN_RATIO,
																test_size=1 - TRAIN_RATIO,
																shuffle=True)
			prep = DataPreProcessor(X_train)
			self.learner.fit(prep.processTweets(X_train, y_train), y_train)
			gc.log(f'Done preproping {train_size} items')
			y_train_pred = self.learner.predict(prep.processTweets(X_train))
			y_test_pred = self.learner.predict(prep.processTweets(X_test))
			train_accuracy[i] = accuracy_score(y_train, y_train_pred)
			test_accuracy[i] = accuracy_score(y_test, y_test_pred)

		self.reportMat(X_train, y_train, prep)
		self.reportMat(X_test, y_test, prep)
		plt.plot(size_range, train_accuracy, label='Train')
		plt.plot(size_range, test_accuracy, label='Test')
		plt.legend()
		gc.save_plt(learner_name)

	def reportMat(self, X, y, prep):
		y_pred = self.learner.predict(prep.processTweets(X))
		print(confusion_matrix(y, y_pred))
		print(accuracy_score(y, y_pred))