#Description: This Program detects if an email is spam(1) or not spam(0)

#Import Libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

#Importing the dataset
df = pd.read_csv('spam_or_not_spam.csv')


#Getting the shape
df.shape

df['label'].value_counts()


#Check for duplicates and remove them
df.drop_duplicates(inplace = True)

df.shape

#Checking for missing data
df.isna().sum()

df = df.dropna()

#Download stopwords package
nltk.download('stopwords')

#Processing the text
def process_text(text):
  
  #1 Remove punctuations
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  #2 Remove stopwords
  clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  #3 Return clean words
  return clean_words

#Tokenization(lemmas)
df['email'].head().apply(process_text)


#Convert collection of text into matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer = process_text).fit_transform(df['email'])

#Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['label'], test_size = 0.2, random_state = 0)

#Create the ML model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Evaluating the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
X_train_pred = classifier.predict(X_train)
print(classification_report(y_train,X_train_pred))
print("Confusion Matrix\n")
print(confusion_matrix(y_train, X_train_pred))
print()
print("Accuracy = {}".format(accuracy_score(y_train, X_train_pred)))

#Predicting the test results
y_pred = classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred))
print()
print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))
print("AUCROC = {}".format(roc_auc_score(y_test, y_pred)))
