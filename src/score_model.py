# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:50:57 2021

@author: sergi
"""

import joblib

X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')
classifier = joblib.load('classifier.pkl')

print(f"Accuracy: {classifier.score(X_test, y_test) * 100:.3f}%", )
