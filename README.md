# Sentiment-Analysis-System

Sentiment Analysis Using Naïve Bayes and Random Forest Classifier


Please read these instructions to run the code:

1: Requirements and Installation

In order to run this code, you will need python3 with following libraries installed
	1: pip install pandas
	2: pip install numpy
	3: pip install nltk
	     Open python terminal by typing python3 in the command prompt and run the following commands
	     >>> import nltk
		>>> nltk.download('stopwords')


2: Running the program
	
	python3 analysis.py -m [mode] -f [filename] -c [classifier type]
	where, 
		mode: 'a' for finding the accuracy and 's' for analysing the reviews
		filename: test file path (csv format) which contains test reviews for the program to analyse
		classifier type: 'm' for Multinomial and 'r' for Random Forest classifier

Note: A training dataset is provided, kindly run the program from the same directory where the training dataset reside. Also, a sample test file is provided as part of the submission.

