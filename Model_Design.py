# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:03:51 2019

@author: dipen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib auto


#In this challenge, we invite Kagglers to help us 
#identify which customers will make a specific transaction in the 
#future, irrespective of the amount of money transacted. The data provided for 
#this competition has the same structure as the real data we have available to 
#solve this problem.

# to get seeding value
seed_value =1902

#path="D:\DS\Wine Data\Sander_Customer_Transaction"
path="D:\me\ds\wine data\WineData\Sander_Customer_Transaction"
os.chdir(path)
print(os.getcwd())

# read the csv file
dataTrain = pd.read_csv("Data/train.csv")
dataTest = pd.read_csv("Data/test.csv")

dataTrain.head()
print(dataTrain.shape)
print('columns: ',dataTrain.columns)

# check the target data
# from data we can see the data is higly skewed
# so lets get the half and half target valued data set
dataTrain.dtypes
dataTrain['target'].value_counts().plot(kind='bar')


# extract all the targeted data set first as 0 and 1 value
dataTrain_1 = dataTrain[dataTrain['target']==1]
print(dataTrain_1.shape)

dataTrain_0 = dataTrain[dataTrain['target']==0]
print(dataTrain_0.shape)

20098/179902
# sample data from train0 data

dataTrain_0_10percent = dataTrain_0.sample(frac=len(dataTrain_1)/len(dataTrain_0)+0.01, random_state=seed_value)
print(dataTrain_0_10percent.shape)

# combined the balanced data
dataTrainBalanced = pd.concat([dataTrain_1,dataTrain_0_10percent], axis=0)
print('shape: ', dataTrainBalanced.shape)

dataTrainBalanced['target'].value_counts().plot(kind='bar')

#check the null value exist or not
df_null_count =dataTrainBalanced.isnull().sum()
df_null_count.value_counts().plot(kind='bar')


# check datatype and see if its int or float or categorical
dataTrainBalanced.dtypes

dataTrainBalanced.head()

# see stastical value
# first check the distributeion
dataTrainBalanced['var_0'].hist( bins =25)
dataTrainBalanced[['var_0']].boxplot()
dataTrainBalanced['var_0'].describe()



#split target and feature 
# no null value at all....cool
dataY = dataTrainBalanced["target"]
dataY.head()
dataX = dataTrainBalanced.drop(["target"], axis=1)

# check the data type of all columns
# it seems there is no correlation at all
import seaborn as sns
sns.pairplot(dataX.iloc[:,1:5])

#clean up the memory for large data
#del dataTrain
#del dataTest
#del dataTrain_0
#del dataTrain_1
#del dataTrain_0_10percent

plt.hist(dataX.iloc[:,1:10])
d = dataX.iloc[:,1:10]
dataX.iloc[:,1:10].hist()
# check data distribution and see if there is any fishy data
plt.ioff()
for i in range(0,1):#len(dataX.columns):
    print(i)
    plt.hist(dataX.iloc[:,i*20:(i+1)*20],bins=20)
    plt.savefig("figure/fig"+1)
    #plt.hist(dataX.iloc[:,1:10],bins=20)
    
plt.ion()





