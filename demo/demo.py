# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 00:21:45 2021

@author: sergi
"""


import joblib
import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer

classifier = joblib.load('classifier.pkl')
classify = dill.load(open('classify.pkl', 'rb'))
predict = dill.load(open('predict.pkl', 'rb'))


print('Type some text to be classified.')
text = input('Text: ')
print()

classify(text, classifier=classifier)
print('Probability of ' + str(int(max(predict(text)[0])*10000)/10000))
