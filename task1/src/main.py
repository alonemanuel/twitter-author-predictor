from baseline import *
from preprocessor import *

def main(X, y):
	prep = Preprocessor()
	X_train, y_train, X_test, y_test = prep.train_test_split(X, y)
	baseliner = Baseline(X_train, y_train)
	
if __name__ == '__main__':
	main()
