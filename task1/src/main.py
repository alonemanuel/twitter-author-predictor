from src.baseline import*
from src.preprocessor import *
from src.poly_regression import PolyRegression
from src.garcon import Garcon
from src.preproccesData import TweetsPreProcessor
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
	X_train, y_train, X_vault, y_vault = prep.train_test_split(X, y)
	X_train, y_train, X_test, y_test = prep.train_test_split(X_train, y_train)

	tweets_proccesor = TweetsPreProcessor()
	proccesed_data = tweets_proccesor.processTweets(X_train)

	# baseliner = Baseline(X_train, y_train)
	# baseliner.report(X_test, y_test)
	# X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
	# # Different learning models
	# models = [PolyRegression]
	# learners = [None]
	# for i, model in enumerate(models):
	# 	learners[i] = model(X_train, y_train)	# Training is done uon creation
	# 	learners[i].report(X_test, y_test)


if __name__ == '__main__':
	main()
