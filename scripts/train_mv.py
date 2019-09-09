from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
import sys
import os
train=[]
test=[]
linstt=[]
#read data from the csv file
dataset = csv.reader(open('../data/mturk_sample.csv'))
testset = csv.reader(open('../data/test.csv'))
for i in dataset:
    train.append(i)
for j in testset:
    test.append(j)
#converting to matricx    
train=np.array(train)
test=np.array(test)
train_set_x = train[1:-1:,2:1203]
test_set_x = test[1:-1:,1:-1]
test_set_y=test[1:-1:,-1]

linstt=train_set_x.tolist()
nnnew=[]
trains=[]
labels=[]
first_ele=[]
new_all=[]
#this is for get the majority data 
for ins in linstt:
    first_ele.append(ins[2])
#create a set for all unique data 
list_data=list(set(first_ele))
#caculate the labels for positive and netgitive
for j in range(len(list_data)):
    countPos=0
    countNeg=0
    c=0
    #if it's eaqual to positive then the positive should be add 1
    #if it's equal to negitive then the negitive should be add 1
    
    for i in range(len(linstt)):
        if linstt[i][2]==list_data[j]:
            c=i
            if linstt[i][-1]=='pos':  
               countPos+=1
            else:
                countNeg+=1   
    #get the real label through the majority vote
    if countPos>countNeg:
        trains.append(linstt[c][0:1200])
        labels.append('pos')
    else:
        trains.append(linstt[c][0:1200])
        labels.append('neg')

trains=np.array(trains)
labels=np.array(labels)     
#decision tree classifier
clf = DecisionTreeClassifier(min_samples_split=100,random_state=0)
#training data
clf.fit(trains,labels)
#get the prediction for data
predicted = clf.predict(test_set_x)
#get the probability
probability = clf.predict_proba(test_set_x)
#get the accuracy
accuracy = accuracy_score(test_set_y,predicted)*100
#get the f_score
f_score = f1_score(test_set_y,predicted,average = "binary",pos_label = "pos")
#draw the condusion matricx
confusion = confusion_matrix(test_set_y, predicted, labels=None, sample_weight=None)
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
