import regex
import re
import emoji
import pandas as pd


class TweetsPreProcessor:

    def processTweets(self, tweets):
        """
        Gets the tweets, spits the data
        :param tweets:
        :return: data_from_tweets - pandas df shape : (num_of_tweets, num_of_features)
        """
        number_of_emoji_per_tweet = []
        number_of_hastags_per_tweet = []
        number_of_mentions_per_tweet = []
        link_exist_per_tweet = []
        all_hashtags = [] #np.empty(shape=(tweets.size, ))
        proccesed_tweets = [] #np.empty(shape=(tweets.size, ))
        all_mentions = []
        for tweet in tweets:
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
            #
            proccesed_tweets.append(tweet)

        ##
            # at this point tweet variable is a cleaned tweet.
            # mentions, hastags, and links can be found in the lists

        data_from_tweets = pd.DataFrame()
        data_from_tweets['Emoji Count'] = number_of_emoji_per_tweet
        data_from_tweets['Hastag Count'] = number_of_hastags_per_tweet
        data_from_tweets['Mention Count'] = number_of_mentions_per_tweet
        data_from_tweets['Link exist'] = link_exist_per_tweet

        return data_from_tweets


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


    def extractHashtags(self, tweet):
        """
        Gets a tweet, return a list of it's hastags (no #) and a tweet with the
        hastags removed. list can be empty if no hastags found
        """
        hash_tags = re.findall(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)", tweet)

        # remove hastags
        for hash_tag in hash_tags:
            tweet = tweet.replace("#" + hash_tag, '')
        return tweet, hash_tags

    def extractMentions(self, tweet):
        """
        Gets a tweet, return a list of it's mentions (no @) and a tweet with
        the mentions removed. list can be empty if no mentions found
        """
        # mentions = re.findall(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", tweet)
        mentions = re.findall(r"@([^\s]+)", tweet)

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
