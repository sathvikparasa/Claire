# predict hit songs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import gdown
from math import sqrt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)

import warnings
warnings.filterwarnings("ignore")

data_path =  './spotify_data_urls.csv'
gdown.download('https://drive.google.com/uc?id=1MIkOcP2JY_foloYAR5-Y60YyRVbRhQMs', data_path, True)

data = pd.read_csv(data_path)
basic_data = data[['Artist','Track','Year','url','Label']]

# basic_data.head(10)
# print(basic_data['Track'][3], ": ", basic_data['url'][3])
# print(basic_data['Track'][1], ": ", basic_data['url'][1])
# print(data.columns.tolist())
# plt.hist(data["danceability"], color = 'b')
# plt.plot()
# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
# ax0.hist(data["danceability"], color = 'b')
# ax0.set_title('Danceability Histogram')
# ax1.hist( , color = 'b') # uncomment these
# ax1.set_title()
# ax2.hist( , color = 'r')
# ax2.set_title()
# ax3.hist( , color = 'r')
# ax3.set_title()
# fig.tight_layout()
# plt.show()

quantitative = data[['Artist', 'Track', 'key', 'mode', 'tempo']] # FILL IN QUANTITATIVE FEATURES
# print(quantitative.head(10))

qualitative = data[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']] # FILL IN FEATURES HERE
# print(qualitative.head(10))

outputs = data[['Artist','Track','Label']]
# print(outputs.head(10))

X = data[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]
y = data[['Label']]

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Logistic Regression Model
# lr = LogisticRegression()
# lr.fit(X_train, y_train) # with splitted training data and testing data

# print(lr.score(X_test, lr.predict(X_test)))

# # coef = lr.coef_
# # print(pd.DataFrame(list(zip(X, coef[0])), columns=['feature', 'coef']))

# # # X_2 = [['danceability', 'loudness', 'mode', 'tempo']] # Whatever features you'd like!
# # # y_2 = data[['Label']]
# # # lr2 = LogisticRegression()
# # # lr2.fit(X_train, y_train)

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(int(sqrt(len(X_train)))),
    SVC(kernel="linear", C=0.025),
    GaussianNB()
]

for classifier in classifiers:
    print("---------------")
    print(str(classifier) + '\n')
    classifier.fit(X_train, y_train)
    print("Accuracy: ", metrics.accuracy_score(y_test, classifier.predict(X_test)))
    print("Precision: ", metrics.precision_score(y_test, classifier.predict(X_test)))
    print("Recall: ", metrics.recall_score(y_test, classifier.predict(X_test)))