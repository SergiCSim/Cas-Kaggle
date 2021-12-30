# Cas Kaggle Aprenentatge Computacional UAB 2021-22
### Nom: Sergi Cantón Simó
### DATASET: Twitter and Reddit Sentimental analysis Dataset
### URL: [kaggle]https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset

## Dataset
In fact, there are two datasets. One of those datasets has text posts of Reddit, each of them labeled with three possible numbers: -1, 0 or 1. The label is -1 if a negative sentiment is expressed, 0 if a neutral sentiment is expressed and 1 if a positive sentiment is expressed. The other datset is formed by tweets from Twitter, labeled the same way as Reddit's.

## Experiments


## Preprocessed
Before starting creating our model, we have done some changes in our datasets. First, we have merged both datasets in one, but keeping Reddit and Twitter datasets. We have, also, dropped all rows with null values.

## Model
Only one model has worked properly.
Model				| Hiperparameters	| Accuracy 	| Time
-----------------------------------------------------------------
KNN					| Default			| 40.5% 	| 0.1 s
SVC					| Polynomial kernel | None		| Infinity
Logistic regression	| Default			| None		| Infinity
Random forest		| Default			| None		| Infinity
XG Boost			| Default			| None		| Infinity
Linear SVC			| Default (C = 1)	| 94.5%		| 8 s

## Demo
 We can test our model using *comanda*

## Conclusions
Our best model has been the linear SVC with default paramers, that is, C = 1. We have got an accuracy of approximately 94.5%.