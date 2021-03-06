# Cas Kaggle Aprenentatge Computacional UAB 2021-22
### Nom: Sergi Cantón Simó
### DATASET: Twitter and Reddit Sentimental analysis Dataset
### URL: [kaggle]https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset

## Datasets
There are two datasets. One of those datasets has text posts of Reddit, each of them labeled with three possible numbers: -1, 0 or 1. The label is -1 if a negative sentiment is expressed, 0 if a neutral sentiment is expressed and 1 if a positive sentiment is expressed. The other datset is formed by tweets from Twitter, labeled the same way as Reddit's.

## Objective/goal
Our objective is to make a classifier to know automatically if a text message expresses a negative, neutral or positive sentiment.

## Experiments
Before starting creating our model, we have made some changes in our datasets. First, we have merged both datasets into one, but keeping Reddit and Twitter datasets. We have, also, dropped all rows with null values.

Then, we have done a first analyisis of our datasets, plotting how many samples are with each label in each sample. He have also done a word cloud for each dataset to visualize which words appear the most.

Once we have had an idea of what our data are like, we have built our classifier using TF-IDF matrices. After splitting data in train and test, we have tried to train six models. Finally, we have chosen a linear SVC model and we have obtained its best C parameter using grid search.

Later, we have tested our model by classifying some texts.

In the end, we have done some extra analyisis. We have searched the most important words to classify by label, and we have plotted precision-recall and ROC curves and we have computed also the confusion matrix of our test data.

## Model
| Model | Hiperparameters | Accuracy | Time |
| -- | -- | -- | -- |
| KNN | Default | 40.5% | 0.1 s |
| SVM | kernel: polynomial | None | Infinity |
| Logistic regression | Default | None | Infinity |
| Random forest | Default | None | Infinity |
| XG Boost | Default | None | Infinity |
| Linear SVM | Default (C: 1) | 94.5% | Infinity |
| -- | -- | -- | -- |

## Demo
We can test our model using *python demo.py*.

## Conclusions
We have reached quite good accuracies. Furthermore, we have tested our classifier giving it some self-made text messages and it has classified our messages well. So we have checked by ourselves that it works properly.

Nevertheless, there are some things that we coud improve in the future. For example, we could use some other libraries like TensorFlow to carry out our classification in a different way or using another model. To generalize more our model, we could also use messages from other social networks, like comments from Instagram, Wallapop or opinion blogs.
