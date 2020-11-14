# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:03:05 2019

@author: Erdo
"""

#%%
""" Homework - Iris Dataset """


# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% 
""" Data Import """
data = pd.read_excel('Iris.xls') 
data.head()

# %% 
""" Categoric Data to Numeric Data """
#Label encoding
IrisColumn = data.iloc[:,-1:].values #numpy array olarak aldık.

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

IrisColumn[:,0] = le.fit_transform(IrisColumn[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

IrisColumn = ohe.fit_transform(IrisColumn).toarray()

IrisColumn = pd.DataFrame(data=IrisColumn, index=range(150), columns=['setosa','versicolor', 'virginica'])


# %% 
""" Get X and Y """

X = data.iloc[:,0:4]

Y1 = IrisColumn.iloc[:,0:1] # Is Iris-setosa?
Y2 = IrisColumn.iloc[:,1:2] # Is Iris-versicolor?
Y3 = IrisColumn.iloc[:,3:4] # Is Iris-virginica?


# %% 
""" Train and Test Set Split """
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Y1train, Y1test = train_test_split(X,Y1,test_size=0.33, random_state=0) 
Xtrain, Xtest, Y2train, Y2test = train_test_split(X,Y2,test_size=0.33, random_state=0)
Xtrain, Xtest, Y3train, Y3test = train_test_split(X,Y3,test_size=0.33, random_state=0)


#%%
""" Feature Scaling """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

Xtrain_std = sc.fit_transform(Xtrain)
Xtest_std = sc.fit_transform(Xtest)

from sklearn import preprocessing 
mms = preprocessing.MinMaxScaler()
Xtrain_nrm = mms.fit_transform(Xtrain) 
Xtest_nrm = mms.fit_transform(Xtest) 

#%%
""" LOGISTIC REGRESSION """
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr3 = LogisticRegression()

lr1.fit(Xtrain,Y1train)
print("Test accuracy {}", format(lr1.score(Xtest,Y1test)))

