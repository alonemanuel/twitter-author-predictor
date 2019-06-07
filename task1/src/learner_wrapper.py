from sklearn.metrics import confusion_matrix, accuracy_score
from task1.src.garcon import Garcon
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gc = Garcon()


class LearnerWrapper:
	'''
	Wraps a sklearn learner implementing the classic ML algorithm API (fit, predict, ...).
	'''


	def __init__(self, model, prep, name):
		'''
		:param prep: 		preprocessor
		'''
		gc.enter_func()
		self.prep = prep
		gc.log(name)
		self.model = model()
		self.classifier_name = name


	def fit(self, X_train, y_train):
		'''
		:param X_train:		type=df,	shape=(n_tweets, )
		:param y_train: 	type=df,	shape=(n_tweets, )
		'''
		gc.enter_func()
		X_train = self.prep.get_tweets_features(X_train)
		gc.log_shape(X_train=X_train)
		gc.log('Done preping')
		y_train = y_train.values
		self.model.fit(X_train, y_train)


	def predict(self, X):
		'''
		Predicts a label vector.
		:param X:	type=df,		shape=(n_tweets, )
		:return: 	type=np.array,	shape=(n_tweets, )
		'''
		gc.enter_func()
		prep_X = self.prep.get_tweets_features(X)
		return self.model.predict(prep_X)


	def report(self, X, y, log='', is_test=False):
		'''
		Reports some data regarding the learner and its performance.
		:param X:	type=df,		shape=(n_samples, )
		:param y: 	type=np.array,	shape=(n_samples, )
		:param log: text to be logged
		:return: 	some info on the screen.
		'''
		gc.enter_func()
		y_pred = self.predict(X)
		if log:
			print(log)
		cm = confusion_matrix(y, y_pred)
		accu_score = '{0:.3f}'.format(accuracy_score(y, y_pred))
		cm_df = pd.DataFrame(cm)
		title_suffix = 'Test set' if is_test else 'Train set'
		gc.init_plt(f'{self.classifier_name}, {title_suffix}\nAccuracy: {accu_score}')
		sns.heatmap(cm_df, annot=True)
		plt.xlabel('True label')
		plt.ylabel('Predicted label')
		fn_suffix = 'testset' if is_test else 'trainset'
		fn_learner_name = self.classifier_name.replace(' ', '')
		gc.save_plt(f'{fn_learner_name}_{fn_suffix}')

def report_predictor(predictor, pred_name, X, y, log='', is_test=False):
	'''
	Reports some data regarding the learner and its performance.
	:param X:	type=df,		shape=(n_samples, )
	:param y: 	type=np.array,	shape=(n_samples, )
	:param log: text to be logged
	:return: 	some info on the screen.
	'''
	gc.enter_func()
	y_pred = predictor.predict(X)
	if log:
		print(log)
	cm = confusion_matrix(y, y_pred)
	accu_score = '{0:.3f}'.format(accuracy_score(y, y_pred))
	cm_df = pd.DataFrame(cm)
	title_suffix = 'Test set' if is_test else 'Train set'
	gc.init_plt(f'{pred_name}, {title_suffix}\nAccuracy: {accu_score}')
	sns.heatmap(cm_df, annot=True)
	plt.xlabel('True label')
	plt.ylabel('Predicted label')
	fn_suffix = 'testset' if is_test else 'trainset'
	fn_learner_name = pred_name.replace(' ', '')
	gc.save_plt(f'{fn_learner_name}_{fn_suffix}')