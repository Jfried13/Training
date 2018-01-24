from random import randint
import numpy as np
import pandas as pd  # used for data processing
from sklearn.feature_extraction.text import CountVectorizer  # sklearn is for machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# REading both of the csv files into datafields
df1 = pd.read_csv('tcss455/training/profile/profile.csv', index_col=0)
df2 = pd.read_csv('tcss455/training/relation/relation.csv', index_col=False)
# a = df2.groupby(['like_id']).agg(['count'])
a = df2.groupby(['like_id']).size().sort_values(ascending=False)
user_data = df1.loc[:, ['userid', 'gender']]
user_data = user_data.set_index('userid')
like_data = df2.loc[:, ['userid', 'like_id']].set_index('like_id')
user_id_list = []
for index, row in user_data.iterrows():
    user_id_list.append(index)
user_gender = df1.loc[:, 'gender'].values
fifty_plus_likes = []

for index_val, series_val in a.iteritems():
    if series_val > 50:
        fifty_plus_likes.append(index_val)
    # print('index = ', index_val, ', series_val ', series_val)

# The likes dict is created by loopoing through any number of times and then
# creating a random number between 0 and 4591, which is the number of pages that
# have received at least 51 or more likes from users.  Then using the randomly chosen
# like_id get all of the user_ids that liked that page and append that to a list and finally
# insert the like_id as the key and the list of userids as the value.
likes_dict = dict()
randomNum = []
for count in range(10):
    idx = randint(0, 4591)
    while idx in randomNum:
        idx = randint(0, 4591)
    page_id = fifty_plus_likes[idx]
    tempDF = df2.loc[df2['like_id'] == page_id]
    user_ids = []
    for index, row in tempDF.iterrows():
        user_ids.append(row['userid'])
    # print(user_ids)
    likes_dict[page_id] = user_ids

useful_like_pages = []
for key, value in likes_dict.items():
    total = 0
    count = 0
    for v in value:
        count = count + 1
        total = total + user_data.loc[v].values[0]
        average = total / count
    if average > .70 or average < .30:
        useful_like_pages.append(key)

finalDF = pd.DataFrame()

# This loops through a list of like_ids and then gets a list of all the pages that
# liked the page.  Then looping through the order of user_ids if the user_id is in the
# list of ids that liked the page then append a 1 meaning that user liked the page, otherwise
# append a zero.  The columncounter is used to decide where to insert a new row into the dataframe
columnCounter = 0
for page in useful_like_pages:
    like_list = []
    uses_that_liked_page = like_data.loc[page].values
    # print(uses_that_liked_page)
    for user in user_id_list:
        if user in uses_that_liked_page:
            like_list.append(1)
        else:
            like_list.append(0)
    # print(like_list)
    finalDF.insert(loc=columnCounter, column=float(page), value=like_list)
    columnCounter = columnCounter + 1
# print(finalDF)
finalDF.insert(loc=0, column='userid', value=user_id_list)
finalDF.insert(loc=len(finalDF.columns), column='gender', value=user_gender)
print(finalDF)

X = finalDF.loc[:, finalDF.columns != 'gender']
print('x=', X)
Y = finalDF.gender  # selecting extraversion as the target
print('y=', Y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)
clf = DecisionTreeClassifier()
clf.fit(X, Y)
print("Accuracy: %.2f" % metrics.accuracy_score(y, clf.predict(X)))

'''from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn import metrics'''

'''
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))    

predicted = cross_val_predict(clf, X, y, cv=20)

print("Accuracy: %.2f" % metrics.accuracy_score(y, predicted))
'''
