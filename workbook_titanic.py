# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:40:57 2020

@author: DevanshiKulshreshtha
"""

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
import numpy as np #foundational package for scientific computing
import scipy as sp #collection of functions for scientific computing and advance mathematics

import sklearn #collection of machine learning algorithms
#misc libraries
import random
import time

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Configure Visualization Defaults
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
pd.set_option('display.max_columns', 100)


# =============================================================================
# Loading Data
# =============================================================================

data_raw = pd.read_csv('D:/Python tutorial/Kaggle Competitions/Titanic/train.csv')
data_val = pd.read_csv('D:/Python tutorial/Kaggle Competitions/Titanic/test.csv')

#creating a copy of the data
data1 = data_raw.copy(deep = True)

#passing both training and test data to one, so cleaning can happen for both
data_cleaner = [data1, data_val]


#preview data
print (data_raw.info()) 
data_raw.head() 
data_raw.tail() 

data1.describe()

# checking data for null values
print(data1.isnull().sum())
print(data_val.isnull().sum())

# age, embarrked and fare have missing values

## DATA CLEANING
#for dataset in data_cleaner:    
#    #complete missing age with median
#    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
#
#    #complete embarked with mode
#    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
#
#    #complete missing fare with median
#    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
        

# Creating a function to standardise fare
from sklearn.preprocessing import FunctionTransformer
def standard_fare(x):
    y = (x - data1['Fare'].mean())/(data1['Fare'].std())
    return y


# Create transformer
fare_transformer = FunctionTransformer(standard_fare)   

# FEATURE ENGINEERING
for dataset in data_cleaner:    
        #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    dataset['Sex_Code'] = LabelEncoder().fit_transform(dataset['Sex'])
    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['Title'].replace(dict.fromkeys(['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir'],
           'Mr'), inplace = True)
    dataset['Title'].replace(dict.fromkeys(['Dona', 'Lady', 'the Countess', 'Jonkheer'],
           'Mrs'), inplace = True)
    dataset['Title'].replace(dict.fromkeys(['Mme', 'Mlle', 'Miss'], 'Ms'), inplace = True)
    
    dataset.loc[((dataset['Title'] == 'Dr') & (dataset['Sex'] == 'male')), 'Title'] = 'Mr'
    dataset.loc[((dataset['Title'] == 'Dr') & (dataset['Sex'] == 'female')), 'Title'] = 'Mrs'

    dataset['Surname'] = dataset['Name'].str.split(", ", expand=True)[0]
    dataset['FamilyID'] = dataset['Surname'] + dataset['FamilySize'].astype(str) 
    dataset.loc[dataset['FamilySize'] <= 2, 'FamilyID'] = 'Small'
    
    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    
    # Replacing Mssing Age cases with median of Title group
    dataset.loc[((dataset.Title == 'Mr') & (dataset.Age.isnull())), 'Age'] = dataset[dataset.Title == 'Mr'].Age.median()
    dataset.loc[((dataset.Title == 'Ms') & (dataset.Age.isnull())), 'Age'] = dataset[dataset.Title == 'Ms'].Age.median()
    dataset.loc[((dataset.Title == 'Mrs') & (dataset.Age.isnull())), 'Age'] = dataset[dataset.Title == 'Mrs'].Age.median()
    dataset.loc[((dataset.Title == 'Master') & (dataset.Age.isnull())), 'Age'] = dataset[dataset.Title == 'Master'].Age.median()
    
    # encoding the title
    dataset['Title_Code'] = LabelEncoder().fit_transform(dataset['Title'])

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
    
    # Age-Class interaction
    dataset['Age*Class']=dataset['Age']*dataset['Pclass']
    
    # Gender-Class interaction
    dataset['Sex*Class']=dataset['Sex_Code'].replace(0, 2)*dataset['Pclass']
    
    dataset['Age_Squared'] = dataset['Age']*dataset['Age']
    dataset['Age*Class_Squared'] = dataset['Age*Class']*dataset['Age*Class']
    
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['IsAdult'] = dataset['Age'] > 18
    
    #Cabin Known/Unknown
    dataset['CabinKnown'] = dataset['Cabin'].notnull()
    
    # Calculating fare per ticket
    dataset['Fare_Individual'] = dataset['Fare']/dataset['FamilySize']
    
    # Standardising the Fare
    dataset['Fare_Standard'] = fare_transformer.transform(dataset['Fare'])
    
    # extracting ticket prefix
    dataset['Ticket_Prfx'] = np.where(dataset['Ticket'].str.split(' ', expand = True)[0].str.isdigit(), 'unknown', 
     dataset['Ticket'].str.split(' ', expand = True)[0].str.replace('\.', ''))



# =============================================================================
# EDA
# =============================================================================

        
        
f, axes = plt.subplots(2, 3)

sns.boxplot(x= "Fare", data=data1,  orient='v' , ax=axes[0, 0])
sns.boxplot(x= "Age", data=data1,  orient='v' , ax=axes[0, 1])
sns.boxplot(x= "FamilySize", data=data1,  orient='v' , ax=axes[0, 2])
sns.countplot(x="FareBin", hue="Survived", data=data1, dodge = False, ax=axes[1, 0])
sns.countplot(x="AgeBin", hue="Survived", data=data1, dodge = False, ax=axes[1, 1])
sns.countplot(x="FamilySize", hue="Survived", data=data1, dodge = False, ax=axes[1, 2])

for ax in f.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)


# =============================================================================
# Modelling
# =============================================================================
    
v_logit =   ['Pclass', 'Sex_Code', 'Fare', 'Fare_Individual', 'Title_Code', 
             'Age*Class', 'Sex*Class', 'Age', 'FamilySize', 'Age_Squared', 
             'Age*Class_Squared']  

from sklearn.linear_model import LogisticRegression

# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0, solver = 'newton-cg', 
                                         penalty = 'none')

# Train model
model = logistic_regression.fit(data1[v_logit], data1.loc[:, 'Survived'])

data_val['Predictions_logit'] = model.predict(data_val[v_logit])

data4 = data_val[['PassengerId', 'Predictions_logit']]  
data4.to_csv('D:/Python tutorial/Kaggle Competitions/Titanic/predictions.csv', index = False)  
    