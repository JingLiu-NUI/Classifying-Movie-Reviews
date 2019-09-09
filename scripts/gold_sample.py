import random
import csv
import pandas as pd
mturk_dataset=csv.reader(open('gold.csv'))
dataset=[]
dataset2=[]
#using random.sample to get the 1000 data
for i in mturk_dataset:
       dataset.append(i)
for j in range(len(dataset)):
       if j!=0:
         dataset2.append(dataset[j])
gold_sample=random.sample(dataset2,1000) 
#convert to dataframae adn save as csv
aa=pd.DataFrame(gold_sample)
aa.to_csv('gold_sample.csv',index=None)