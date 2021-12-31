# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:44:21 2021

@author: sergi
"""

import joblib
from sklearn.svm import LinearSVC

X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')


# Train Linear SVC
classifier = LinearSVC()
classifier.fit(X_train, y_train)


joblib.dump(classifier, 'classifier.pkl')
