from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sys
import os
# read the data from gold_sample.csv
dataset = pd.read_csv('../data/gold_sample.csv')
# read the data from test.csv
testset = pd.read_csv('../data/test.csv')
# get the column from 1 to -1 to get the training and test data
train = list(dataset.columns[1:-1])
test = list(testset.columns[1:-1])
# seprate the training data  and label, as 1201 is the label
train_set_x = dataset.loc[:,train]
train_set_y = dataset.loc[:,"1201"]
# the same application as above
test_set_x = testset.loc[:,test]
test_set_y=testset.loc[:,"class"]
# call “DecisionTreeClassifier” set the attributs as 1000 and 0 in 
# order to get the prediction probability
clf = DecisionTreeClassifier(min_samples_split=1000,random_state=0)
# use fit function to train dataset
clf.fit(train_set_x,train_set_y)
# call function "predict_proba" to get the prediciton probability
probability = clf.predict_proba(test_set_x)
# get predict labels
predicted = clf.predict(test_set_x)
# call function to get accuraacy
accuracy = accuracy_score(test_set_y,predicted)*100
# call function to get f1-score
f_score = f1_score(test_set_y,predicted,average = "binary",pos_label = "pos")
# call function to get confusion matrix
confusion = confusion_matrix(test_set_y, predicted, labels=None, sample_weight=None)
#plot the comfusion matrix
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

print('prediciotn of gold sample are:',predicted)
print('prediciotn  probability of gold sample are:',probability)
print('accuracy of gold sample are:',accuracy)
print('Fi score of gold sample are:',f_score)


