#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:05:06 2018

@author: haykel
"""

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

Data = np.genfromtxt('MWMOTEsampled.csv', delimiter=',')
x_train = Data[1:Data.shape[0]:5, 1:30]
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
y_train = Data[1:Data.shape[0]:5, 30]

Data_test = np.genfromtxt('testData.csv', delimiter=',')
x_test = Data_test[:, 1:30]
x_test = scaler.transform(x_test)
y_test = Data_test[:, 30]

clf = svm.SVC(C=1.0, cache_size=1000)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

print(f1_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(precision_score(y_test, y_predict))