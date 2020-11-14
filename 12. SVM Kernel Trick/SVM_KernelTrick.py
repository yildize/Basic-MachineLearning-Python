# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:23:42 2019

@author: Erdo
"""


"""
--> Diyelim ki elimizde linear bir doğruyla ayrılamayacak bir data var. Bunu ayırmak istiyoruz:
--> Öncelikle soft-margin(hard-margin olursa margin içinde hiç data kalmasına izin vermez) linear kernel kullanarak bazı data pointlerin margin arasında kalmasına göz yumabiliriz.

--> Burada kernel trick Andrew'den farklı anlatılmış
--> Gaussian function ile sanki bir alanı 3. boyuta veya n+1. boyuta çekip,
--> Data sonra bu ayrılan kısmın linear bir hyperplane ile ayrılabileceğinden bahsediyor.

--> Belki Andrew'in bahsettiği de aynı kapıya çıkıyor olabilir.

--> Sonuçta non-linear decision boundry elde etmemizi sağlıyor.

--> Aslında Andrew'inki de böyle düşünülebilir en nihayetinde 
--> Her training data noktasına gaussianlar atıp hangilerinin katsayısı yüksek hangilerinin düşük
--> Onu belirler ve daha sonra bir non-linear region'a boyut artışı gerçekletiririz ve buraya düşen datayı
--> Artık ayırabiliriz.

"""

"""
BUNUN UYGULAMASI SVM ile aynı SADECE KULLANILAN FONSKIYONU DEĞİŞTİRİYORUZ:
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
""" APPLY SVM """

from sklearn.svm import SVC

svc = SVC(kernel="rbf") # Linear SVM kullanılıyor. Başka fonksiyonlar da kullanılabilirdi. Linear SVM log. reg. ile çok benzerdi. Andrew notlarına bakarbilirsin.
svc.fit(X_train, Y_train)

#Prediction
y_pred = svc.predict(X_test)

#Accuracy
print("Test accuracy {}", format(svc.score(X_test,Y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)









