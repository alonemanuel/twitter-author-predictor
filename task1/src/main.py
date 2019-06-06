from task1.src.garcon import Garcon
from task1.src.learners import *

from task1.src.baseline import *
from task1.src.learners.decision_tree import DecisionTree
from task1.src.learners.random_forest import RandomForest
from task1.src.learners.l_d_a import LDA
from task1.src.learners.knn import KNN
from task1.src.learners.learner_analyzer import LearnerAnalyzer
from task1.src.learners.logistic_regression import LogisticReg
from task1.src.preprocessor import *

from sklearn.pipeline import Pipeline


gc = Garcon()


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
	models = [LogisticReg]#, RandomForest, DecisionTree, LDA]#KNN
	learners = [None, None, None]
	for i, model in enumerate(models):
		print("\n\nPerformance of " + str(model.__name__) + " model:\n\n")
		analyzer = LearnerAnalyzer(model, X_train, y_train)
		analyzer.report()


if __name__ == '__main__':
	main()
