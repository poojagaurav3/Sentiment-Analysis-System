import re
import sys
import argparse
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Loading training data file
trainData = "trainingData.csv"

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="mode - accuracy (a) or analyze sentiment (s)", required=True)
parser.add_argument("-f", help="input test file path")
parser.add_argument("-c", help="classifier type - Multinomial Bayes (m) or Random Forest (r)")

args = parser.parse_args()

if(args.m =='s'):
    if(args.f == None or args.c == None or len(args.f) == 0 or len(args.c)==0):
        print("Input file path and classifier type are required ")
        sys.exit()

wordTranf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

#Cleaning Data
def cleanData(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#','', text) #removing # tags
    text = re.sub(r'RT[\s]+', '', text) #removing Retweets
    text = re.sub(r'https?:\/\/\S+', '', text)#removing hyperlinks
     
    return text

def getStringArray(dataframe):
    list=[]
    for wd in dataframe.values:
        if len(str(wd[0]))>0:
            list.append(cleanData(str(wd[0])))
    return list

def processData(mode, filePath, classifier):
    allTweets_X = pd.read_csv(trainData, usecols=[10], header=None, dtype='unicode')
    trainingLabels = pd.read_csv(trainData, usecols=[1], header=None, dtype='unicode')
    trainingDataVector = wordTranf.fit_transform(getStringArray(allTweets_X))

    if (mode == 'a'):
        getAccuracy(trainingDataVector, trainingLabels)
    else:
        getSentimentfromData(filePath, classifier, trainingDataVector, trainingLabels)

def getAccuracy(wordTfidf, polarity_Y):
    #Dividing the data into training and test set
    intrainData, intestSet, cls_trainData, cls_testSet = train_test_split(wordTfidf, polarity_Y.values.ravel(), test_size=0.2, random_state=0)

    # Multinomial NB
    mnbClassifier = getClassifier('m')
    mnbClassifier.fit(intrainData, cls_trainData)

    predTweetsData = mnbClassifier.predict(intestSet)
   
    print("\n--------------------------------------\n")
    print("Multinomial Naive Bayes\n")
    print("--------------------------------------\n")
    print("\nConfusion Matrix: \n", confusion_matrix(cls_testSet,predTweetsData))  
    print("\nClassification report: \n", classification_report(cls_testSet,predTweetsData))  
    print("\nAccuracy score: \n", accuracy_score(cls_testSet, predTweetsData))

    # Random Forest
    rfClassifier = getClassifier('r')
    rfClassifier.fit(intrainData, cls_trainData)

    predTweetsData = rfClassifier.predict(intestSet)
   
    print("\n\n--------------------------------------\n")
    print("Random Forest\n")
    print("--------------------------------------\n")
    print("\nConfusion Matrix: \n", confusion_matrix(cls_testSet,predTweetsData))  
    print("\nClassification report: \n", classification_report(cls_testSet,predTweetsData))  
    print("\nAccuracy score: \n", accuracy_score(cls_testSet, predTweetsData))

def getSentiment(text, classifier):
    nwordTfidf = wordTranf.transform([cleanData(text)])
    predData = classifier.predict(nwordTfidf)

    for x in predData:
        if (x == 'positive'):
            return("Positive tweet,%s\n"%(text))  
        elif(x == 'negative'):
            return("Negative tweet,%s\n"%(text))

    return("Neutral tweet,%s\n"%(text))
  
def getSentimentfromData(inputPath, classifier, trainingDataVector, trainingLabels):
    # Train
    print("Training classifier...")
    classifier = getClassifier(classifier)
    classifier.fit(trainingDataVector, trainingLabels.values.ravel())

    print("Training complete")

    # open test data file
    output = open('SentimentAnalaysis.csv', 'w')
    output.write("Polarity,Review\n")
    with open(inputPath, errors = 'ignore') as fileO:
        for line in fileO:
            try:
                str=getSentiment(line, classifier)
                output.write(str)
            except:
                pass

    output.close()

    print("Output file:'SentimentAnalaysis.csv'")

def getClassifier(classifier):
    if (classifier == 'm'):
        return MultinomialNB()
    elif (classifier == 'r'):
        return RandomForestClassifier(n_estimators=100, random_state=0)
    print("Invalid classifier type, please provide 'm' for multinomial and 'r' for random forest classifier")
    sys.exit()
    
processData(args.m, args.f, args.c)



