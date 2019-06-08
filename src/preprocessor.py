import regex
import re
import emoji
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from task1.src.garcon import Garcon

gc = Garcon()

nltk.download('stopwords')

NUM_OF_CLASSES = 10
IS_TFIDF = True
TRAIN_RATIO = 0.5
DATA_DIR_PATH = 'tweets_data'
CSV_EXT = ".csv"


class Preprocessor:
	'''
	Preprocesses the data.
	'''


	def __init__(self, raw_X, raw_y):
		'''
		Ctor for a preprocessor.
		:param tweets:	type=df, shape=(n_samples, 2)
		'''

		self.tweets = raw_X
		self.labels = raw_y
		self.vectorizer = None

		self.common_hashtags = []
		self.common_mentions = []
		self.common_lexemes = []


	def preprocess(self):
		'''
		Begins preprocessing.
		'''
		# self._split_tweets_labels()
		self._set_vectorizer()


	def _split_tweets_labels(self):
		'''
		Splits raw data into tweets and labels.
		'''
		self.tweets = self.raw_data['tweet']
		self.labels = self.raw_data['user']


	def _set_vectorizer(self):
		'''
		Inits a vectorizer for the bag of words.
		'''
		if IS_TFIDF:
			self._tfidf_vectorize()
		else:
			self._count_vectorize()
		self.vectorizer.fit(self.tweets)


	def _tfidf_vectorize(self):
		self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.6, ngram_range=(1, 2), stop_words='english',
										  max_features=25000)


	def _count_vectorize(self, min_ngram=2, max_ngram=3, max_feat=1000, binary=False):
		self.vectorizer = CountVectorizer(stop_words='english', analyzer='word',
										  ngram_range=(min_ngram, max_ngram),
										  max_features=max_feat, binary=binary)


	def get_tweets_features(self, new_tweets):
		'''
		Extracts features from tweets.
		:param new_tweets:	type=df, 		shape=(n_tweets, )
		:return: 			type=np.array, 	shape=(n_tweets, n_features)
		'''
		bow_df = self._get_bow_df(new_tweets)
		indicators_df = self._get_indicators_df(new_tweets)
		features = pd.concat([bow_df, indicators_df], axis='columns')
		return features.values


	def _get_bow_df(self, tweets):
		'''
		:param tweets:	type=df,	shape=(n_tweets, )
		:return: 		type=df,	shape=(n_tweets, n_bags)
		'''
		bow = self.vectorizer.transform(tweets)
		vecd_df = pd.DataFrame(bow.toarray(), columns=self.vectorizer.get_feature_names())
		return vecd_df


	def _get_indicators_df(self, tweets):
		'''
		:param tweets:	type=df,	shape=(n_tweets, )
		:return: 		type=df,	shape=(n_tweets, n_indicators)
		'''
		df = pd.DataFrame()
		n_chars = []
		n_words = []
		n_hashtags = []
		n_mentions = []
		n_links = []
		n_emojis = []

		for tweet in tweets:
			n_chars.append(self.get_n_chars(tweet))
			n_words.append(self.get_n_words(tweet))
			n_hashtags.append(self.get_n_hashtags(tweet))
			n_mentions.append(self.get_n_mentions(tweet))
			n_links.append(self.get_n_links(tweet))
			n_emojis.append(self.get_n_emojis(tweet))

		df['n_chars'] = n_chars
		df['n_words'] = n_words
		df['n_hashtags'] = n_hashtags
		df['n_mentions'] = n_mentions
		df['n_links'] = n_links
		df['n_emojis'] = n_emojis
		return df


	def get_n_words(self, tweet):
		'''
		Return word count for tweet.
		:param tweet:
		:return:
		'''
		return len(tweet.split())


	def get_n_chars(self, tweet):
		'''
		Returns the char count of the tweet (of all chars).
		:return:
		'''
		return len(tweet)


	def get_n_hashtags(self, tweet):
		"""
		Gets a tweet, return a list of it's hastags (no #) and a tweet with the
		hastags removed. list can be empty if no hastags found
		"""
		hash_tags = re.findall(
			r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)",
			tweet)

		return len(hash_tags)


	def get_n_mentions(self, tweet):
		"""
		Gets a tweet, return a list of it's mentions (no @) and a tweet with
		the mentions removed. list can be empty if no mentions found
		"""
		mentions = re.findall(
			r"(?<=^|(?<=[^a-zA-Z0-9]))@([A-Za-z]+[A-Za-z0-9-_]+)",
			tweet)
		return len(mentions)


	def get_n_links(self, tweet):
		"""
		Gets a tweet, return a list of it's links (full link) and a tweet with
		the links removed. list can be empty if no links found
		"""
		links = re.findall(r"http\S+", tweet)
		return len(links)


	def get_n_emojis(self, tweet):
		"""
		Gets a tweet, return number of emojis in the tweet and a tweet with
		the emojis removed.
		"""
		emoji_set = set()
		number_of_emoji = 0
		line = regex.findall(r'\X', tweet)
		for word in line:
			if any(char in emoji.UNICODE_EMOJI for char in word):
				emoji_set.add(word)
				number_of_emoji += 1

		return number_of_emoji
