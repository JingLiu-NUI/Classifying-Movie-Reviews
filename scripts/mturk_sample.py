import random
import csv
import pandas as pd
gold_dataset=csv.reader(open('gold_sample.csv'))
mturk_dataset=csv.reader(open('mturk.csv'))
gold_list=[]
mturk_list=[]
new_line=[]
new_line1=[]
nn=[]
new_line11=[]
#read data from the dataset
#then compare the id with gold_sample, if it's same then keep it
for line1 in gold_dataset:
    new_line.append(line1)
for line2 in mturk_dataset:
    new_line1.append(line2)
for i in new_line:
    for j in new_line1:
        if str(i[0])==str(j[0]):
            new_line11.append(j)

#convert to dataframe and save as csv
aa=pd.DataFrame(new_line11)
aa.to_csv('mturk_sample.csv',index=None)


