# Resheph Gohil (B00238972)
# MSc Big Data
# University of the West of Scotland

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics as mts
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_validation import KFold, cross_val_score

# read machine learning dataset created to be used with scikit learn into a pandas dataframe with name "mlDf"
mlDf = pd.read_csv('data/mldata_scikit.csv')

##################
# Train/Test Split
##################
X = mlDf.drop(['los','time_in_hospital'], axis=1)
y = mlDf.los

X_train, X_test, y_train, y_test = tts(X, y, random_state=40, train_size=0.7, test_size=0.3) # sampling the data within X and y into four seperate sets
X_train = X_train.as_matrix() # transform dataframe into matrix
X_test = X_test.as_matrix() # transform dataframe into matrix

X_train.shape
X_test.shape
y_train.shape
y_test.shape

##########################
# KNN (K-Nearest Neighbor)
##########################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
knn

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_score = mts.accuracy_score(y_test, knn_pred)
knn_score

knn_cross_val_score = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
knn_cross_val_score
knn_cross_val_score.mean()

#finding the right K for KNN
k_range = range(1, 31)
knn_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn , X, y, cv=10, scoring="accuracy")
    knn_scores.append(scores.mean())
print(knn_scores)

kofknn = plt.plot(k_range, knn_scores)
kofknn = plt.xlabel('Value of k for KNN')
kofknn = plt.ylabel('Cross-Validated Accuracy')
kofknn.figure.savefig('graphs_findings/value_of_k_for_knn.png')

###############################
# SVM (Support Vector Machines)
###############################
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=0.01)
svm

svm.fit(X_train, y_train)

# evaluating the data using the predict() command and the test data
svm_pred = svm.predict(X_test)
svm_score = mts.accuracy_score(y_test, svm_pred)
svm_score

svm_cross_val_score = cross_val_score(svm, X, y, cv=10, scoring='accuracy')
svm_cross_val_score
svm_cross_val_score.mean()

#finding the right C for SVM
c_range = [0.001, 0.01, 0.1, 1]
svm_scores = []
for c in c_range:
    svm = SVC(kernel='linear', C=0.01)
    scores = cross_val_score(svm, X, y, cv=10, scoring="accuracy")
    svm_scores.append(scores.mean())
print(svm_scores)

cforsvm = plt.plot(c_range, svm_scores)
cforsvm = plt.xlabel('Value of c for SVM')
cforsvm = plt.ylabel('Cross-Validated Accuracy')
cforsvm.figure.savefig('graphs_findings/value_of_c_for_svm.png')

################
# Decision Trees
################
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree

tree.fit(X_train, y_train)

# evaluating the data using the predict() command and the test data
tree_pred = tree.predict(X_test)
tree_score = mts.accuracy_score(y_test, tree_pred)
tree_score

tree_cross_val_score = cross_val_score(tree, X, y, cv=10, scoring='accuracy')
tree_cross_val_score
tree_cross_val_score.mean()

#finding the right min_sample_split for Decision Trees model
samples_range = [2,4,6,8,10,12,14,16,18,20]
tree_scores = []
for s in samples_range:
    tree = DecisionTreeClassifier(min_samples_split = s)
    scores = cross_val_score(tree, X, y, cv=10, scoring="accuracy")
    tree_scores.append(scores.mean())
print(tree_scores)

dt = plt.plot(samples_range, tree_scores)
dt = plt.xlabel('Value of minimum Sample Split for Decision Trees')
dt = plt.ylabel('Cross-Validated Accuracy')
dt.figure.savefig('graphs_findings/value_of_min_sample_split_for_decision_trees.png')

#####################
# Logistic Regression
#####################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg

logreg.fit(X_train, y_train)

logreg_predict = logreg.predict(X_test)
logreg_score = mts.accuracy_score(y_test, logreg_predict)
logreg_score

logreg_cross_val_score = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
logreg_cross_val_score
logreg_cross_val_score.mean()

#finding the right C for Logistic Regression
c_range = [0.001, 0.01, 0.1, 1]
lr_scores = []
for c in c_range:
    logreg = LogisticRegression(penalty='l2', C = c)
    scores = cross_val_score(logreg, X, y, cv=10, scoring="accuracy")
    lr_scores.append(scores.mean())
print(lr_scores)

coflogreg = plt.plot(c_range, lr_scores)
coflogreg = plt.xlabel('Value of c for logreg')
coflogreg = plt.ylabel('Cross-Validated Accuracy')
coflogreg.figure.savefig('graphs_findings/value_of_c_for_logreg.png')
