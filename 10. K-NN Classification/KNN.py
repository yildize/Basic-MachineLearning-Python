# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:54:34 2019

@author: Erdo
"""

"""
K - Nearest Neighbors

--> Feature Scaling Önemliydi.

--> Basit ve çok güçlü bir algoritmadır.
--> Eğitime gerek yoktur.
--> Yeni gelen data'ya en yakın N tane training set data'sı bulunur ve bu dataların sonucuna göre yeni gelen datanın sonucu tahmin edilir.

--> Sıkıntıları var tabi.
--> Mesela boy ve kilo'ya göre yaş tahmin ediyoruz.
--> 1 birim boy uzaklığı ile 1 birim kilo uzaklığı aynı şey olmuyor.
--> Bu sebeple farklı uzaklık hesaplama algoritmaları kullanılabiliyor.

--> K çift sayı ise mesela 4 ise ve sonuç 2'ye 2 çıkarsa ilgili data point'e en yakın olan 2'li seçilir.

--> Lazy Learning ve Eager Learning gibi çeşitleri vardır. 
--> Bizim şuan bahsettiğimiz lazy learning herhangi bir eğitim olmuyor.
--> Eager learning ise önce bölgeleri hazırlar ve training data'yı unutur. Daha sonra yeni datanın hangi bölgeye düştüğüne bakar.

"""

"""
Logistic Regression için Kullanılan Sign Language Örneğini Kullanalım:
    
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
""" APPLY K-NN """

from sklearn.neighbors import  KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski') #5 komşuya baksın ve minkowski mesafesini kullansın.
knn.fit(X_train,Y_train)

y_pred = knn.predict(X_test)

print("Test accuracy {}", format(knn.score(X_test,Y_test)))


"""
KESİN BİR KURAL YOK AMA K SEÇİMİ İÇİN GENEL CONVENTION:
    
    K = ((m_training)^0.5)/2
       
"""



#%%
""" Confusion Matrix: """

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)


"""
KNN ile NON-LINEAR DECISION BOUNDARIES ELDE EDİLEBİLİR. 
LOG. REG. GIBI LINEARLY SEPERABLE DATA OLMASI GEREKMIYOR.
YANI DATA'YA BAKIP ONA GORE ALGORITMA BELIRLEMEK LAZIM.


AYRICA N'I ARTIRMAK HER ZAMAN İYİ SONUÇ VERMEZ.
MESELA DATA'DA KÜÇÜK GRUPLANMALAR VAR İSE N KÜÇÜLDÜKÇE DAHA İYİ SONUÇ ALIRIZ.

"""


