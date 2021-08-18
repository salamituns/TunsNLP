# load data
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from nltk import corpus
import pandas as pd
sms = pd.read_csv('tunsNLP/smsspamcollection/SMSSpamCollection',
                  sep=('\t'), names=['labels', 'message'])

# data preprocessing

ps = PorterStemmer()
corpus = []

for i in range(len(sms)):
    # remove and replace other words with space, excluding a-z & A-Z
    review = re.sub('[^a-zA-Z]', ' ', sms['message'][i])
    # convert to lower case
    review = review.lower()
    # splitting the words
    review = review.split()

    # filter out the stopwords and stem the remaining words
    review = [ps.stem(word) for word in review if word is not set(
        stopwords.words('english'))]
    # then join
    review = ' '.join(review)
    # and append to corpus
    corpus.append(review)

# convert labels to dummy variable
y = pd.get_dummies(sms['labels'])
y = y.iloc[:, 1].values

# implement BoW from sklearn using the countvectorizer library
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

# import test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

# train models using Naive Bayes classifier
model = MultinomialNB().fit(X_train, y_train)

# testing the model to predict
y_pred = model.predict(X_test)

# check for the confusion matrix
confusion = confusion_matrix(y_pred, y_test)
confusion

# check model accuracy
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
print(confusion)
