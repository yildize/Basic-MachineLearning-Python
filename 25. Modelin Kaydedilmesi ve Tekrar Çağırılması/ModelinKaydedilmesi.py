# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:57:18 2019

@author: Seko
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:53:06 2019

@author: Erdo
"""

#%%
""" MESELA BİR MODELİ EĞİTMEMİZ 1 HAFTA SÜRDÜ
    TEKRAR AÇTIĞIMDA YENİDEN Mİ EĞİTECEĞİZ?
    HAYIR TABİ Kİ PICKLE LIBRARY KULLANCAĞIZ:
"""
"""
Bir önceki örnekteki XGBoost classifier'ını kaydetmeyi ve çağırmayı görelim:
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
""" XGBoost """
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train) # Sonuçta bu bir classifier. XGBoostta bir classification alg. demekki.

y_pred = classifier.predict(x_test)

#%%
""" MODELİN KAYDEDİLMESİ """
import pickle

dosya = "model.kayit"
pickle.dump(classifier,open(dosya,'wb'))

#%%
""" MODELİN TEKRAR ÇAĞIRILMASI """
yuklenen = pickle.load(open(dosya,'rb'))
print(yuklenen.predict(x_test))





