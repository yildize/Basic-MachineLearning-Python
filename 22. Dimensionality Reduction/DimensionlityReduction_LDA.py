# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:27:30 2019

@author: Erdo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:58:56 2019

@author: Erdo
"""

""" Linear Discriminant Analysis: 2. Dimensionality Reduction Algoritmamız"""
""" PCA ile çok benzer, LDA supervised."""
""" Yani classları hesaba katarak bunları en iyi şekilde ayıracak yeni dimensionları verir. """
""" ÖNCE PCA kodunu aynen yazdım sonra LDA'e geçtik. İki Algoritmayı karşılaştıracağız """
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

from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 2 adet Principal Components elde edeceğiz dedik.
X_train2 = pca.fit_transform(X_train) #X_train üzerinden hem reduction yapacak yeni boyutları bulacak hemde üzerine X_train verilerini dönüştürecek ve X_train2 içine kaydedecek.

X_test2 = pca.transform(X_test)



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


#%% 
""" LDA APPLICATION """
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train) #Yeni eksenleri fit ettik ve LDA ile X_train'i transform ettik.
X_test_lda = lda.transform(X_test) #X_test'i de yeni eksenlere transform ettik.


#%%
""" Tekrar bir Log. Reg. tanımlayalım """
#Principal Components kullanılarak eğitilen Log Reg:
classifier3 = LogisticRegression(random_state=0)
classifier3.fit(X_train_lda,y_train)

#Predict
y_pred3 = classifier3.predict(X_test_lda)

#LDA sonrası vs Original data
cm4 = confusion_matrix(y_pred3,y_pred)
print(cm4)

""" SONUÇTA YİNE %100 BAŞARI SAĞLADIK. 
    PCA İLE 1 VERİ FİRE VERMIŞTİK
    LDA SINIFLARIN ETİKETİNE DE DİKKAT ETTİĞİ İÇİN PERFORMANSI ARTIRDI.
"""