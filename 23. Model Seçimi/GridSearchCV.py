# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:17:23 2019

@author: Erdo
"""

#%%
""" MODEL SEÇİLDİKTEN SONRA, MODEL İÇİNDE OPTİMİZE EDİLECEK BİR ÇOK DEĞER VAR BUNLARI NASIL OPTİMİZE EDERİZ?

--> Bunun için GridSearchCV kullanacağız.
--> Geçen örnekte kullanılan SVC'yi kullanacağız.
--> SVC algoritması içinde bir çok parametre var: C, kernel tipi , gamma, etc...

"""

#%%
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
#%%
# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#%%
# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#%%
# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
#%%
# Tahminler
y_pred = classifier.predict(X_test)
#%%
#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#%%
#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
''' 
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı

'''
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print(basari.mean())
print(basari.std())

#Yani X_train ve y_train'i açıkladığımız gibi 4 kez bölüyor ve her biri için classifier'ı eğitiyor.
#daha sonra test ediyor.
#sonuç veriyor.

#std bize verilerin ne kadar birbirinden farklı çıktığını verir.
#%%
""" GRIDSEARCHCV APPLICATION """

#Parametre optimizasyonu ve algoritma seçimi:
from sklearn.model_selection import GridSearchCV

#Şimdi biz deneyeceğimiz parametreleri bir list of dictionary olarak verilecek ve daha sonra bunlar
#classifier içinde tek tek denenecek ve en iyi kombinasyon seçilecek.

p = [{'C':[1,2,3,4,5],'kernel':['linear']}, #linear için 5 kombinasyon dener.
     {'C':[1,10,100,1000],'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]} #rbf için 20 kombinasyon denenir.
     ]

# Burada önce dev adımlar atıp sonra küçük adımlar atmak mantıklı.
# Mesela C 100 ile 1000 arasına düşerse bu aralıkda daha küçük adımlarla sonuç tekrar denenebilir.

"""
GS PARAMETERS
estimator: hangi algoritmayı optimize etmek istediğimiz (objeyi veriyoruz)
param_grid: denenmesini istediğimiz parametreler.
scoring: neye göre scorelanacağı, accuracy olabilir, precision olabilir, recall olabilir, purity etc.
cv: Kaç folding olacağı. Sadece GridSearch kullanırsak bu folding olmuyor sanırım.
n_jobs: aynı anda çalışacak iş 
"""

gs =  GridSearchCV(estimator = classifier, param_grid=p, scoring='accuracy', cv=10, n_jobs=-1 )


grid_search = gs.fit(X_train, y_train)  # altta sürekli SVM classifier çalışacak, eğitilecek, sonuçlar ölçülecek.

bestScore = grid_search.best_score_
bestParameters = grid_search.best_params_

print(bestScore)
print(bestParameters) 
# Sonuçta en iyi parametreler ve score bulunmuş oldu.







