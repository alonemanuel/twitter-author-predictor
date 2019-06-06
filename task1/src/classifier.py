import pickle

def classify(tweets):
	'''
	:param tweets:	samples to tag. shape=(n_samples, )
	:return:	y_hat = labels (0-9) of samples. shape=(n_samples, )
	'''
	classifier_f = open('classify-final-build.pickle', 'rb')
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	return classifier.predict(tweets)
