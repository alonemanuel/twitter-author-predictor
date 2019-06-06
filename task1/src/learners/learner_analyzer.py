from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from task1.src.garcon import Garcon
import numpy as np
from sklearn.model_selection import train_test_split

gc = Garcon()
TRAIN_RATIO = 0.8
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
		size_range = np.arange(5000, max_train_size, STEP_SIZE)
		train_accuracy = np.zeros(size_range.shape)
		test_accuracy = np.zeros(size_range.shape)
		h=None
		for i, train_size in enumerate(size_range):
			new_X = self.X_train[:train_size]
			new_y = self.y_train[:train_size]
			X_train, X_test, y_train, y_test = train_test_split(new_X, new_y,
																train_size=TRAIN_RATIO,
																test_size=1 - TRAIN_RATIO,
																shuffle=True)
			gc.log(f'Done preproping {train_size} items')
			h = self.learner(X_train, y_train)
			y_train_pred = h.classify(X_train)
			y_test_pred = h.classify(X_test)
			train_accuracy[i] = accuracy_score(y_train, y_train_pred)
			test_accuracy[i] = accuracy_score(y_test, y_test_pred)
		h.report(X_train, y_train)
		h.report(X_test, y_test)
		plt.plot(size_range, train_accuracy, label='Train')
		plt.plot(size_range, test_accuracy, label='Test')
		plt.legend()
		gc.save_plt(learner_name)