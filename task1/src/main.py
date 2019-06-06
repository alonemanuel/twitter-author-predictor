from task1.src.garcon import Garcon
from task1.src.learners import *

from task1.src.baseline import *
from task1.src.learners.decision_tree import DecisionTree
from task1.src.learners.knn import KNN
from task1.src.learners.learner_analyzer import LearnerAnalyzer
from task1.src.learners.logistic_regression import LogisticReg
from task1.src.preprocessor2 import *

gc  = Garcon()


def main():
	'''
	Runs everything, and in particular creates a class of hypotheses and
	creates a report for each.
	'''
	gc.enter_func()
	# Helps in preprocessing tasks
	prep = Preprocessor()
	X, y = prep.get_train_data()
	X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
	# Different learning models
	# models = [LogisticReg]
	models = [DecisionTree, LogisticReg, KNN]
	learners = [None, None, None]
	for i, model in enumerate(models):
		analyzer = LearnerAnalyzer(model, X_train, y_train)
		analyzer.report()


if __name__ == '__main__':
	main()