# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:01:58 2019

@author: Seko
"""

#%% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
data = pd.read_csv('maaslar.csv') 
data.head()

# %% 
""" Obtain the training set x and y """

x = data.iloc[:,1:2] #eğitim seviyesini numeric olarak aldık.
y = data.iloc[:,2:] #maasları aldık.

# %% 
""" Training set'i görselleştirelim """
plt.scatter(x,y,color='red')
# Göründüğü üzere burada polynomial bir input-output ilişkisi var bu ilişki linear regression ile iyi modellenemez.
# Fakat zaten daha önceden bildiğimiz gibi input x'in çeşitli orderlarını lr ile kullanarak polynomial reg elde edebiliriz.
# Theta0 + Theta1*x + Theta2*x^2 + Theta3*x^3 gibi...

# %% 
""" Linear Regression Building the Model """

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor.fit(x,y)
y_pred = lin_regressor.predict(x)

#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred,color='blue')


# %% 
"""Polynomial Regression Building the Model (DİKKAT ET DAHA ÖNCE LR İÇİN XO=1 EKLEMEMİŞTİK FAKAT BURADA EKLEDİK DEMEKKİ EKLENİYOR) """

from sklearn.preprocessing import PolynomialFeatures
polyFeatureModel = PolynomialFeatures(degree=4) # x0, x1, x1^2  şeklinde featureları oluşturacak modeli elde ettik. 
polyFeatures = polyFeatureModel.fit_transform(x) # polyFeatures içinde artık ilgili featureları tutar.

# BU NOKTADAN SONRA ARTIK POLY FEATURES numpy array olarak HAZIR OLDUĞUNA GÖRE LİNEAR REGRESSOR İLE POLY MODEL KURULABILIR.
lin_regressor2 = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor2.fit(polyFeatures,y) #polynomial regressor gibi düşünülebilir.
y_pred = lin_regressor2.predict(polyFeatures) #polynomial regressor'ın tahminleri

#plot the polynomial final hypothesis
plt.scatter(x,y,color='red') #normal data pointler
plt.plot(x,y_pred,color='blue') #polynomial regressor'ın tahminleri.


# %% 
""" Single data prediction"""
#Linear Regression Model
print(lin_regressor.predict([[11]]))  #input array (veya lists of list) formunda olmalı.
print(lin_regressor.predict([[6.6]]))

#Polynomial Regression Model - 3. derece olduğu için 3 feature input girmeli:
polyFeatures1 = polyFeatureModel.fit_transform([[11]])
polyFeatures2 = polyFeatureModel.fit_transform([[6.6]])

print(lin_regressor2.predict(polyFeatures1))
print(lin_regressor2.predict(polyFeatures2))
