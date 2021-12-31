# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 11:27:47 2021

@author: sergi
"""

import joblib
import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.model_selection import train_test_split

compute_tfidf_matrix = dill.load(open('compute_tfidf_matrix.pkl', 'rb'))


# Load Reddit data
reddit_data = pd.read_csv('reddit_Data.csv')
reddit_data.rename(columns = {'clean_comment': 'text'}, inplace = True)

# Load Twitter data
twitter_data = pd.read_csv('Twitter_Data.csv')
twitter_data.rename(columns = {'clean_text': 'text'}, inplace = True)

# Combine both dataframes into one dataframe: data
data = pd.concat([reddit_data, twitter_data], ignore_index = True)

# Drop NaN's in each dataset
reddit_data.dropna(axis = 0, inplace = True)
twitter_data.dropna(axis = 0, inplace = True)
data.dropna(axis = 0, inplace = True)

# Reset dataset's indices (they have been damaged due the dropping)
data = data.reset_index()

# Get X and y from the combined dataset
X = compute_tfidf_matrix(data['text'])
y = data.category

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# To save our matrices
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
