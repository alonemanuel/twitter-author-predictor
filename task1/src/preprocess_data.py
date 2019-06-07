import regex
import re
import emoji
from collections import defaultdict
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import functools as ft
import numpy as np

nltk.download('stopwords')

import langdetect

NUM_OF_CLASSES = 10


class DataPreProcessor:
    def __init__(self, tweets):
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000,
                                          binary=False)
        self.most_common_hashtags = []
        self.most_common_mentions = []
        self.most_common_lexemes = []
        self.english = []
        self.french = []
        self.spanish = []
        self.italic = []
        self.portuguese = []
        self.deutsch = []

        self.vectorizer.fit(tweets)

    def processTweets(self, tweets, labels=None):
        """
        Gets the tweets, spits the data
        :param tweets:
        :return: data_from_tweets - pandas df shape : (num_of_tweets, num_of_features)
        """
        self.english = []
        self.french = []
        self.spanish = []
        self.italic = []
        self.portuguese = []
        self.deutsch = []
        number_of_emoji_per_tweet = []
        number_of_hastags_per_tweet = []
        number_of_mentions_per_tweet = []
        word_counts_per_tweet = []
        link_exist_per_tweet = []
        char_count_per_tweet = []
        all_hashtags = []
        proccesed_tweets = []
        all_mentions = []

        for tweet in tweets:
            tweet = tweet.replace("\n", "")
            # word count
            word_counts_per_tweet.append(self.get_word_count(tweet))

            # char count
            char_count_per_tweet.append(self.get_char_count(tweet))

            # filter hastags (#)
            tweet, hashtags = self.extractHashtags(tweet)
            number_of_hastags_per_tweet.append(len(hashtags))
            all_hashtags.append(hashtags)

            # filter mentions (@)
            tweet, mentions = self.extractMentions(tweet)
            number_of_mentions_per_tweet.append(len(mentions))
            all_mentions.append(mentions)

            # filter urls (https://...)
            tweet, links = self.exractLinks(tweet)
            link_exist_per_tweet.append(1 if links else 0)

            # filter emoji
            tweet, number_of_emoji = self.extractEmoji(tweet)
            number_of_emoji_per_tweet.append(number_of_emoji)

            self.getLang(tweet)

            # lexemes preproccesing
            tweet = self.normalize_lexical_words(tweet)

            proccesed_tweets.append(tweet)

        ##
        # at this point tweet variable is a cleaned tweet.
        # mentions, hastags, and links can be found in the lists

        data_from_tweets = pd.DataFrame()
        data_from_tweets['Emoji Count'] = number_of_emoji_per_tweet
        data_from_tweets['Hashtag Count'] = number_of_hastags_per_tweet
        data_from_tweets['Mention Count'] = number_of_mentions_per_tweet
        data_from_tweets['Link exist'] = link_exist_per_tweet
        data_from_tweets['Word count'] = word_counts_per_tweet
        data_from_tweets['Char count'] = char_count_per_tweet

        data_from_tweets['English'] = self.english
        data_from_tweets['Italian'] = self.italic
        data_from_tweets['French'] = self.french
        data_from_tweets['Deutsch'] = self.deutsch
        data_from_tweets['Portuguese'] = self.portuguese
        data_from_tweets['Spanish'] = self.spanish

        if labels is not None:
            self.most_common_hashtags = self.get_most_common_words(
                all_hashtags, labels)
            self.most_common_mentions = self.get_most_common_words(
                all_mentions, labels)
            self.most_common_lexemes = self.get_most_common_words(
                proccesed_tweets,
                labels, 100)

        hashtags_for_person = np.zeros((len(tweets), 10))
        mentios_for_person = np.zeros((len(tweets), 10))
        lexems_for_person = np.zeros((len(tweets), 10))

        for index, tweet in enumerate(tweets):
            for hashtag in all_hashtags[index]:
                for i, common_per_person in enumerate(self.most_common_hashtags):
                    hashtags_for_person[index, i] += self.count_word_in_list_of_tuples(common_per_person, hashtag)

            for mention in all_mentions[index]:
                for i, common_per_person in enumerate(self.most_common_mentions):
                    mentios_for_person[index, i] += self.count_word_in_list_of_tuples(common_per_person, mention)

            for lexem in proccesed_tweets[index]:
                for i, common_per_person in enumerate(self.most_common_lexemes):
                    lexems_for_person[index, i] += self.count_word_in_list_of_tuples(common_per_person, lexem)

        for i in range(10):
            name = 'hashtags %f' % i
            data_from_tweets[name] = hashtags_for_person[:, i]

            name = 'mentions %f' % i
            data_from_tweets[name] = mentios_for_person[:, i]

            name = 'lexems %f' % i
            data_from_tweets[name] = lexems_for_person[:, i]


        bow = self.vectorizer.transform(tweets).toarray()
        features = np.column_stack((data_from_tweets.values, bow))
        return features

    def count_word_in_list_of_tuples(self, lst, value):
        """

		:return:
		"""
        for (word, times) in lst:
            if word == value:
                return 1
        return 0

    def normalize_lexical_words(self, tweet):
        '''

		:param tweet:
		:return:
		'''

        stemmer = PorterStemmer()
        words = stopwords.words("english")
        df = pd.Series(tweet).apply(lambda x: " ".
                                    join([stemmer.stem(i) for i in
                                          re.sub("[^a-zA-Z]", " ",
                                                 x).split()]).lower())
        filtered = df.values.tolist()[0].split()

        filtered = [i for i in filtered if i not in words]
        return filtered

    def get_most_common_words(self, words, labels, num_of_common=10):
        """

		:param words: list of list of (hash)tags for each tweet
		:param labels: labels of the tweets in the same order
		:return: list of lists: list[i] contains 5 most common tags for labels[i]
		"""
        most_common = [defaultdict(int) for i in range(NUM_OF_CLASSES)]
        # creates a dictionary for each label containing a word and the times of her appeariance
        for (words_lst, label) in zip(words, labels):
            for word in words_lst:
                most_common[label][word] += 1

        # creating an intersection of all the word's sets
        intersections = [dic.keys() for dic in most_common]
        intersections = ft.reduce(lambda d1, d2: set(d1) & set(d2),
                                  intersections)

        # for each label filter the words that are not in the intersection
        for label, dic in enumerate(most_common):
            most_common[label] = list(
                filter(lambda x: x[0] not in intersections, dic.items()))

        # for each label sort it's vale by thier appreance in decending order
        for i, lst in enumerate(most_common):
            most_common[i] = sorted(lst, key=lambda x: x[1], reverse=True)[
                             :num_of_common]

        return most_common

    def extractEmoji(self, tweet):
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

        # remove emojis
        for emoji_type in emoji_set:
            tweet = tweet.replace(emoji_type, '')
        return tweet, number_of_emoji

    def get_word_count(self, tweet):
        '''
		Return word count for tweet.
		:param tweet:
		:return:
		'''
        return len(tweet.split())

    def get_char_count(self, tweet):
        '''
		Returns the char count of the tweet (of all chars).
		:return:
		'''
        return len(tweet)

    def extractHashtags(self, tweet):
        """
		Gets a tweet, return a list of it's hastags (no #) and a tweet with the
		hastags removed. list can be empty if no hastags found
		"""
        hash_tags = re.findall(
            r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)",
            tweet)

        # remove hastags
        for hash_tag in hash_tags:
            tweet = tweet.replace("#" + hash_tag, '')
        return tweet, hash_tags

    def extractMentions(self, tweet):
        """
		Gets a tweet, return a list of it's mentions (no @) and a tweet with
		the mentions removed. list can be empty if no mentions found
		"""
        mentions = re.findall(
            r"(?<=^|(?<=[^a-zA-Z0-9]))@([A-Za-z]+[A-Za-z0-9-_]+)",
            tweet)

        # remove mentions
        for mention in mentions:
            tweet = tweet.replace("@" + mention, '')
        return tweet, mentions

    def exractLinks(self, tweet):
        """
		Gets a tweet, return a list of it's links (full link) and a tweet with
		the links removed. list can be empty if no links found
		"""
        links = re.findall(r"http\S+", tweet)
        return re.sub(r"http\S+", "", tweet), links

    def getLang(self, tweet):
        """

        :param tweet:
        :return:
        """
        if len(tweet) < 5:
            lang = None
        else:
            try:
                lang = langdetect.detect(tweet)
            except:
                lang = None
        self.english.append(lang == "en")
        self.italic.append(lang == "it")
        self.portuguese.append(lang == "pt")
        self.deutsch.append(lang == "de")
        self.spanish.append(lang == "es")
        self.french.append(lang == "fr")