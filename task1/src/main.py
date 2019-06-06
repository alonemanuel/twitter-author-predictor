from sklearn.linear_model import LogisticRegression

from task1.src.garcon import Garcon
from task1.src.learners import *
from task1.src.learners.abstract_learner import LearnerWrapper

from task1.src.learners.decision_tree import DecisionTree
from task1.src.learners.knn import KNN
from task1.src.learners.learner_analyzer import LearnerAnalyzer
from task1.src.learners.logistic_regression import LogisticReg
from task1.src.preprocessor import Preprocessor
from task1.src.preprocessor2 import Preprocessor2

gc = Garcon()

def main():
	'''
	Runs everything, and in particular creates a class of hypotheses and
	creates a report for each.
	'''
	gc.enter_func()
	# Helps in preprocessing tasks
	file_prep = Preprocessor2()
	X, y = file_prep.get_train_data()
	X_train_raw, y_train_raw, X_test_raw, y_test_raw = \
		file_prep.train_test_split(X, y)
	data_prep = Preprocessor(X_train_raw, y_train_raw)
	data_prep.preprocess()

	linreg = LearnerWrapper(LogisticRegression, data_prep, 'Linear Regression')
	linreg.fit(X_train_raw, y_train_raw)
	linreg.report(X_train_raw, y_train_raw, log='Train:')
	linreg.report(X_test_raw, y_test_raw, log='Test:')

	# Different learning models
	# models = [DecisionTree, LogisticReg, KNN]
	# models = [LogisticReg]
	# learners = [None, None, None]
	# for i, model in enumerate(models):
	# 	analyzer = LearnerAnalyzer(data_prep)
	# 	analyzer.report()

if __name__ == '__main__':
	main()
