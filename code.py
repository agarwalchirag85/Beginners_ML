# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:49:00 2019

@author: Chirag
"""

# importing libraries

import pandas as pd                   #For data analysis
import numpy as np                    # For mathematical calculations
import matplotlib.pyplot as plt       # For ploting graphs

# loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Let's check the features present in our data and then we will look at their data types.
train.columns
test.columns


train.shape, test.shape


# Print data types for each variable
train.dtypes

#printing first five rows of the dataset
train.head()

#converting the target catagorical variable into continous variable
train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)

#checking for null (NAN) in the training dataset
train.isnull().sum()

#model Building 

target = train['subscribed']                    #seperating the training dataset target variable
train = train.drop('subscribed',1)              #droping the target variable form the dataset

# applying dummies on the train dataset
train = pd.get_dummies(train)



from sklearn.model_selection import train_test_split
# splitting into train and validation with 20% data in validation set and 80% data in train set.
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=0)

#Logistic Regression


from sklearn.linear_model import LogisticRegression

# defining the logistic regression model
lreg = LogisticRegression()

# fitting the model on  X_train and y_train
lreg.fit(X_train,y_train)

# making prediction on the validation set
prediction = lreg.predict(X_val)

#calculate the accuracy on validation set.

from sklearn.metrics import accuracy_score
# calculating the accuracy score
accuracy_score(y_val, prediction)


#final prediction on test dataset
final_prediction = lreg.predict(test)


