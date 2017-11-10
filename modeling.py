# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:41:05 2017

@author: gyujin
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

df_train_merged.info()


## CV (k-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# split train ,test, target
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df_train_merged, df_train_target, test_size = 0.3)


# Navie Bayes
model_1= GaussianNB()
scoring='accuracy'
score= cross_val_score(model_1, train_data,train_labels,cv=k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)
# RF
model_2 = RandomForestClassifier(n_estimators=250, max_depth=25)
scoring = 'accuracy'
score = cross_val_score(model_2, train_data, train_labels, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)
# Knn
model_3 = KNeighborsClassifier(n_neighbors = 250)
scoring = 'accuracy'
score = cross_val_score(model_3, train_data, train_labels, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# SVM
model_4 = SVC()
scoring = 'accuracy'
score = cross_val_score(model_4, train_data, train_labels, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
