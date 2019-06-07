# ML Hackathon - HUJI

Team members:
Shahar Nahum, Or Mizrahi, Daniel Levin , Alon Emanuel

Files List:

1. save_model.py: The main function in the project is in this file. The main function
parser the data from files, pre-process it and the run all the classifiers on
the processed data in order to fit it. Nevertheless, it also pre-process the
test data and then predict it.

2. preproccessData.py: This file recieves a list of tweets and process them as follows:
	* It takes out the hashtags wrriten in it, count the num of hashtags in it, and makes a 	list of common hashtags per label.
	* It takes out the mentions wrriten in it, count the num of mentions in it, and makes a 	list of common mentions per label.
	* It takes out the emojis wrriten in it and count the num of emojis in it.
	* Then it takes out all the words left in the tweet, it filteres all the general words of 		the languege ( "the", "it", "i" and so on), it transform all the words to their origins 	and count the most common used words for each label, filtering the words that are inside 	the intersection of all the labels common words.

3. classifier.py: run our trained classifier and then the predict function on the test data you 		will supply.

4. data_getter.py: This file reads the train data and divides it to test and train data (0.85% - 			train) and send it to the data proccessor.

5.learner_analyzer.py: prints all the relevant plots.

6.learner_wrapper.py: This fie is a wrapper to our learner. It wraps the fit and predict functions.

7.preprocessor.py: This file is an early attempt to extract fitures from tweets.

8. gracon.py: A debugger class.


