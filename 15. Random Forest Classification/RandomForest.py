# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:39:40 2019

@author: Erdo
"""

#%%
"""
RANDOM FOREST: Daha önce de bahsettik asıl olay Majority Voting. Bir çok decision treenin sonucuna göre seçiyor.

"""

# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% load data set
x_l = np.load('X.npy') # 2062 adet 64x64 image var 
Y_l = np.load('Y.npy') # Her image için 0-9 arası label var 
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[3].reshape(img_size, img_size)) #reshape'e gerek yok bence.
plt.axis('off') #axisler görünmesin diye.
plt.subplot(1, 2, 2)
plt.imshow(x_l[1].reshape(img_size, img_size))
plt.axis('off')

#%% 
"""Önce yalnızca label 0 ve label 1 için bir dataset yaratalım ve bunu ayıran bir regressor yaratalım:"""
""" 204:408 arası 0, 822:1026 arası 1.  Her digit için 205 sample"""

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# Now we have 410 images each is 64x64 px. And 410 labels (0 and 1)

#%%
""" Let's split the data to Train and Test Sets"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
m_train = X_train.shape[0]
m_test = X_test.shape[0]

#%%
""" X'i input olarak verebilmek için 3 boyuttan 2 boyuta çekelim. Her satır bir picture temsil etsin """

X_train = X_train.reshape(m_train,X_train.shape[1]*X_train.shape[2]) # m_train x 4096
X_test = X_test .reshape(m_test,X_test.shape[1]*X_test.shape[2]) # m_test x 4096
print("X train flatten",X_train.shape)
print("X test flatten",X_test.shape)

#%%
""" APPLY RANDOM FOREST """

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, criterion ="entropy")
rfc.fit(X_train,Y_train)

y_pred = rfc.predict(X_test)


#Accuracy
print("Test accuracy {}", format(rfc.score(X_test,Y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)

#%%
""" TAHMİN PROBABILITY'LERİNİ GÖREBİLİRİZ: """
y_proba =  rfc.predict_proba(X_test)
print(y_proba) # Her örnek için olasılık sonuçlarını verir. Kimi örnekleri yüksek olasılıkla tahmin eder kimini düşük.

#%%
""" ROC EĞRİSİNİ KULLANALIM """

from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(Y_test,y_proba[:,0],pos_label=1)


# BU ŞUAN ÇALIŞMIYOR AMA BÖYLE BİR ŞEY GEREKİRSE BAKARSIN

