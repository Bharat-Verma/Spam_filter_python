# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:00:41 2017

@author: Bharat
"""
import pandas as pd

#importing the dataset
dataset = pd.read_csv("combined_emails_main.csv")

#Cleaning the texts
import re
#remove the non significant words which doesnt help us to 
#know whether email is spam or not (e.g. is, am, are etc)
#stopwords is the list of those words.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#stemming the words will make sure to take the root words only
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#corpus is list of all the words for each email
corpus = []
#pre processing the text of the email by removing all the unwanted spaces,
#converting the words in lower case and 
#removing all the words that are in the stopwords list from the email texts
#after this for loop each word in corpus is lower case and root word
for i in range(0,1200):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Email_body'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
#Creating the Bag of Words for all the emails
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
                
#splitting the data into training set and testing set
#train_size parameter can be used to control the number of training emails
#random_state parameter can be used to randomly shuffle the training and testing emails 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.833, random_state = 42)

#Create your NB classifier
#fit the data to the NB classifier for training
from sklearn.naive_bayes import BernoulliNB
classifier_bernoulli = BernoulliNB()
classifier_bernoulli.fit(X_train, Y_train)

#predict the weather the email is ham or spam for test emails
Y_pred_bernoulli = classifier_bernoulli.predict(X_test)

# Making the Confusion Matrix 2x2
# The count of true negatives is CM[0,0], false negatives is CM[1,0], 
# true positives is CM[1,1] and false positives is CM[0,1].
from sklearn.metrics import confusion_matrix
cm_bernoulli = confusion_matrix(Y_test, Y_pred_bernoulli)
print ("Confusion matrix: ")
print (cm_bernoulli) 

#Calculating Accuracy
accuracy_bournalli = (cm_bernoulli[0,0]+cm_bernoulli[1,1])/(cm_bernoulli[0,0]+cm_bernoulli[1,1]+cm_bernoulli[0,1]+cm_bernoulli[1,0])  
print ("Accuracy = "+str(accuracy_bournalli*100)+" %")

#Calculating Precision
precision_bernoulli = (cm_bernoulli[1,1])/(cm_bernoulli[1,1]+cm_bernoulli[0,1])
print ("Precision = "+str(precision_bernoulli*100)+" %")

#Calculating Recall
recall_bernoulli = (cm_bernoulli[1,1])/(cm_bernoulli[1,1]+cm_bernoulli[1,0])
print ("Recall = "+str(recall_bernoulli*100)+" %")

#Calculating F1-Score
F1_score_bernoulli = (2*(precision_bernoulli)*(recall_bernoulli))/((precision_bernoulli)+(recall_bernoulli))
print ("F1-Score = "+str(F1_score_bernoulli*100)+" %")