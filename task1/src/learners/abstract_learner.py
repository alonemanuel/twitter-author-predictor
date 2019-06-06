from sklearn.metrics import confusion_matrix, accuracy_score

from task1.src.garcon import Garcon

gc = Garcon()


class LearnerWrapper():
    def __init__(self, model, prep, name):
        '''
		:param prep: 		preprocessor
		'''
        gc.enter_func()
        self.prep = prep
        self.model = model()
        self.classifier_name = name

    def fit(self, X_train, y_train):
        '''
		:param X_train:		type=df,	shape=(n_tweets, )
		:param y_train: 	type=df,	shape=(n_tweets, )
		'''
        gc.enter_func()
        print(y_train.shape)
        X_train = self.prep.processTweets(X_train)
        y_train = y_train.values
        self.model.fit(X_train, y_train)

    def predict(self, X):
        gc.enter_func()
        prep_X = self.prep.processTweets(X)
        return self.model.predict(prep_X)

    def report(self, X, y, log=''):
        gc.enter_func()
        y_pred = self.predict(X)
        if log:
            print(log)
        print(confusion_matrix(y, y_pred))
        print(accuracy_score(y, y_pred))
