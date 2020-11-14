# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:53:06 2019

@author: Erdo
"""

#%%
"""
      XGBoost: Sonuçta alternatif bir CLASSIFICATION algoritması.

--> XGBoost detaylarını kendi dökümantasyonundan okuyabilirsin.
--> Çok iş yapan deep learning ile çalışan bir algoritma.
--> Yüksek verilerle iyi performans gösteriyor.
--> Hızlı çalışıyor ve hafızayı verimli kullanıyor.
--> Problem ve modelin yorumunun mümkün olması.    


      
--> Derin öğrenme için kullandığımız müşteri kaybetme dataseti kullanılacak:
"""

#%%
#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('Churn_Modelling.csv')
data.head()

#%%
#veri on isleme
X= data.iloc[:,3:13].values
Y = data.iloc[:,13].values

#%%
#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]

#%%
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#%%
#verilerin olceklenmesi
""" FEATURE SCALING UYGULAMAYA GEREK KALMIYOR SANIRIM """

#%% 
""" XGBoost """
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train) # Sonuçta bu bir classifier. XGBoostta bir classification alg. demekki.

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)








