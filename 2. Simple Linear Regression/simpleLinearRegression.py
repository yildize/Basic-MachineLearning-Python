# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:25:52 2019

@author: Erdo
"""

# %% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
salesDF = pd.read_csv('satislar.csv') 

# %% 
""" Obtain Training and Test Sets """
X = salesDF[['Aylar']]
Y = salesDF[['Satislar']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=0) 

# %% 
""" Linear Regression Building the Model """

from sklearn.linear_model import LinearRegression
lr = LinearRegression() # bir LinearRegression object tanımladık.

lr.fit(x_train,y_train) # lr objesi training set'i alarak eğitim gerçekleştirir. Cost function'ı minimize eden model parametrelerine ulaşır.

# %% 
""" Linear Regression Application """
prediction = lr.predict(x_test) #Eğitilmiş model verilen inputa karşın çıktıları üretir.

# %% 
""" Simple Linear Regression Görselleştirme """

# training data'yı çizdirmek istiyorum ama test datası ayırmak için sırasını karıştırmıştık önce sort etmeliyiz:
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.scatter(x_train,y_train)
plt.plot(x_test,prediction) # Bu doğru aslında bizim final hypothesis'imizi gösterir.

plt.title('Aylara göre satış rakamları')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')




