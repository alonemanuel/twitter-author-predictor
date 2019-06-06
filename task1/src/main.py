from learners.baseline import *
from learners.decision_tree import DecisionTree
from preprocessor import *

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
	models = [DecisionTree]
	learners = [None]
	for i, model in enumerate(models):
		learners[i] = model(X_train, y_train)	# Training is done uon creation
		# learners[i].report(X_test, y_test)


if __name__ == '__main__':
	main()
