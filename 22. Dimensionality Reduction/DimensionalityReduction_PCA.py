# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:58:56 2019

@author: Erdo
"""
#%%
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

#%%
# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
""" Amacımız 13 tane feature yerine daha az feature kullanmak ve bununla customer_segmentation'ı tahmin etmek.
-->Yani aynı algoritmayı hem 13 feature için hemde indirgenmiş daha az feature için çalıştırıp sonuçları karşılaştıracağız!
"""
#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 2 adet Principal Components elde edeceğiz dedik.

X_train2 = pca.fit_transform(X_train) #X_train üzerinden hem reduction yapacak yeni boyutları bulacak hemde üzerine X_train verilerini dönüştürecek ve X_train2 içine kaydedecek.
#yani fit ile Ureduced u buluyor.
#transform ile z'yi buluyor.
#fit_transform ile ikisini aynı anda yapıyor.

""" ANDREW'İN DERSLERİNE BAK.
-->Orada göreceksin ki Ureduced matrisini çıkarma işlemi yalnızca X_train üzerinden yapılıyor.
-->Yeni datadan yeni bir Ureduced bulmayacaz sadece çıkarılan eksenlere taşıyacağız.
-->Bu yüzden burada da fit'i birdaha kullanmayacağız sadece transform'u kullancağız.
"""

X_test2 = pca.transform(X_test)

#X_train2 ve X_test2 artık 2 boyutlu.


#%%
""" Şimdi bir Principal Components ve Original Components için Log. Reg. eğitip performans karşılaştıralım.
"""
from sklearn.linear_model import LogisticRegression

#Original features kullanılarak eğitilen Log Reg:
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Principal Components kullanılarak eğitilen Log Reg:
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#Predictions
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

#%%
"""Confusion matrixlere bakalım"""

from sklearn.metrics import confusion_matrix

#original performance
cm = confusion_matrix(y_test,y_pred)
print(cm)

#after PCA performance
cm = confusion_matrix(y_test,y_pred2)
print(cm)
 

#original vs after PCA
cm = confusion_matrix(y_pred,y_pred2)
print(cm)


#Sonuç olarak PCA ile input sayısını çok azalttık bu bize bir hızlanma getirir.
#Bunun yanında performansta ufak bir kayıp yaşadık.
#Bu bir trade-off bu kararı uygulamaya göre veririz.

#Ayrıca bazen PCA başarıyı artıra da bilir.





