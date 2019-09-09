from sklearn.metrics import confusion_matrix
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
import matplotlib.pyplot as plt
inputfile = '../data/mturk_sample.csv'
inputfile2='../data/test.csv'
dataset = pd.read_csv(inputfile, encoding="UTF-8")
movie = dataset["0"]
crowed_label = dataset["1202"]
crowed_name = dataset["1"]
data = pd.DataFrame({"Movie":movie,"CLASS":crowed_label,"NAME":crowed_name })
data.groupby('Movie')["CLASS"].sum()
# the real label of moive which is majotiried by people
m=data.groupby([data['Movie'], data['CLASS']])["NAME"].count().reset_index(name="Count")
# use group ti get the index of movie
Movie = m.groupby('Movie').apply(lambda t: t[t.Count==t.Count.max()]).reset_index(drop=True).transform("Movie")
# use group_by to get the majority label of movie
CLASS = m.groupby('Movie').apply(lambda t: t[t.Count==t.Count.max()]).reset_index(drop=True).transform("CLASS")
new = pd.DataFrame({"Movie":Movie,"CLASS":CLASS})
# use duplicated to get the first label when the number of vote is equal
# eg: 1028 has three positive and three negative, then we get the first one (positive)
real_movie_label = new.drop_duplicates("Movie","first", inplace=False)

# Movie_real_label.groupby([data['Movie'], data['CLASS']]).count()
#change the label of charactor to 1.0 or 0.0
real_movie_label['neg']=real_movie_label['CLASS'].apply(lambda x: 1.0  if x=='neg' else 0.0)
real_movie_label['pos']=real_movie_label['CLASS'].apply(lambda x: 1.0  if x=='pos' else 0.0)
# get the label of movie
pos=real_movie_label.transform("pos")
neg=real_movie_label.transform("neg")
y_matrix= pd.DataFrame({"pos":pos,"neg":neg})
y_matrix=np.array(y_matrix)
y_true = data.groupby([data['Movie'], data['NAME']])["CLASS"].sum().unstack()
y_true = np.array(y_true)
labels=["pos", "neg"]

#this is the update process for DS algorithm
#here I only get 10 iteration
for i in range(10):
    #initialize the confusion matircx for each worker
    C=np.zeros((y_true.shape[1],2,2), dtype=float, order='C')
    #this is the first update for the confusion matrixs,
    for i in range(y_true.shape[1]):
        for j in range(y_true.shape[0]):
            #on the input data table, there are lots of cell are NaN, so
            #here we just use the if the cell are not null, then caculate it
            if pd.isnull(y_true[j][i])==False:  
                #if the input data are positive
                if y_true[j][i]==labels[0]:
                    # if the matricx 0(pos)>matricx1(neg)
                    if y_matrix[j][0]>y_matrix[j][1]:
                        #confusion mat[p,p] +1
                        C[i][0][0]+=1.00
                        #confusion mat[1,0] +0
                        C[i][1][0]+=0.00
                    else:
                        #else like below
                        C[i][0][0]+=0.00
                        C[i][1][0]+=y_matrix[j][1]
                else: 
                    #if the input data is negitive
                    # if the matricx1(neg) >matricx0(pos)
                    if y_matrix[j][1]>y_matrix[j][0]:
                        C[i][0][1]+=0.00
                        C[i][1][1]+=1.00
                    else:
                        # else like below
                        C[i][0][1]+=y_matrix[j][0]
                        C[i][1][1]+=0.00       
    #this is the nomalize for the data in confusion matrix
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
          if C[i][j][0]+C[i][j][1]!=0:
            gap=1/float((C[i][j][0]+C[i][j][1]))
            pos_norm=gap*C[i][j][0]
            neg_norm=gap*C[i][j][1]
            C[i][j][0]=pos_norm
            C[i][j][1]=neg_norm
   #initalize y matrix with 0
    y_matrix=np.zeros(y_matrix.shape, dtype=float, order='C')
    #using the consudion matrics and input data to update y_matricx
    for i in range(y_true.shape[1]):
        for j in range(y_true.shape[0]):
            #still make use the data in input data are not null
            # update the inital majority dataset 
              if pd.isnull(y_true[j][i])==False:
                if y_true[j][i]==labels[0]:
                    y_matrix[j][0]+=C[i][0][0]
                    y_matrix[j][1]+=C[i][1][0]
                else:
                    y_matrix[j][0]+=C[i][0][1]
                    y_matrix[j][1]+=C[i][1][1]
      
    #nomalize the probability in matrix and get from 0-1
    for i in range(y_matrix.shape[0]):
      if y_matrix[i][0]+y_matrix[i][1]!=0:
        gap2=1/float((y_matrix[i][0]+y_matrix[i][1]))
        pos_norm2=gap2*y_matrix[i][0]
        neg_norm2=gap2*y_matrix[i][1]
        y_matrix[i][0]=pos_norm2
        y_matrix[i][1]=neg_norm2
labels=[]    
#get the new label    
for i in range(y_matrix.shape[0]):
    if y_matrix[i][0] > y_matrix[i][1]:
       labels.append('pos')
    else:
       labels.append('neg')          
train=[]
test=[]
linstt=[]
#reasding data
dataset = csv.reader(open(inputfile))
testset = csv.reader(open(inputfile2))
for i in dataset:
    train.append(i)
for j in testset:
    test.append(j)
#cconverting to matricx 
train=np.array(train)
test=np.array(test)
#get the train data and testing dat
train_set_x = train[1:1001,2:1202]
test_set_x = test[1:-1,1:-1]
test_set_y=test[1:-1,-1]
#define the decision three classifier
clf = DecisionTreeClassifier(min_samples_split=100,random_state=0)
#training data
clf.fit(train_set_x,labels)
#predict data and evaluate
predicted = clf.predict(test_set_x)
probability = clf.predict_proba(test_set_x)
# print("probability:",probability)
accuracy = accuracy_score(test_set_y,predicted)*100
# print(len(test_set_y))
# print(len(predicted))
f_score = f1_score(test_set_y,predicted,average = "binary",pos_label = "pos")

confusion = confusion_matrix(test_set_y, predicted, labels=None, sample_weight=None)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

print('prediciotn of ds sample are:',predicted)
print('prediciotn  probability of ds sample are:',probability)
print('accuracy of ds sample are:',accuracy)
print('Fi score of ds sample are:',f_score)