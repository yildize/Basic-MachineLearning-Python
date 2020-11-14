# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:19:14 2019

@author: Erdo
"""

#%%
""" NAIVE BAYES CLASSIFICATION (Anladığım kadarıyla tüm featurelar kategorik olmalı ki bu alg. iyi çalışsın.)

--> Naive Bayes hatırlarsan koşullu olasılıkd denklemini yazan abimiz.
--> P(A|B) =  P(A ∩ B) / P(B)
--> Küme teorisinden düşünülebilir. B yağmur yağma olasılığım gerçekleşti diyelim. A şemsiyemi alma olasılığım?
--> A ile B'nin kesişiminin B ye bölümü. Mantıklı.

--> Bayes Teoremi koşullu olasılığın yerini değiştirmemize olanak sağlar:
    
    P(C|X) = P(X|C) P(C) / P(X) 
    
    AGE     INCOME       STUDENT       CREDIT_RATING      BUYCOMPUTER
    <30      HIGH         NO                FAIR            NO
    <30      LOW          NO                EXCELLENT       NO
    30-40    MEDIUM      YES                FAIR            YES
    
--> Yukarıdaki gibi bir data set olsun.
--> Biz bu dataset'e göre bazı olasılıkları hesaplayabiliriz. 
--> Mesela yaşı <30 olanların bilgisayar alma olasılığı.
--> Yani biz herhangi bir feature'ın sağlanması durumunda bilgisayar alma olasılığını biliyoruz.

--> C bilgisayar alma durumu.
--> X herhangi bir feature için bir class'ın gerçekleşme durumu.

--> Naive Bayes şöyle çalışıyor:
    
    
    Compute P(C): 
        
        *P(buys_computer ="yes") = 9/14 = 0.643
        *P(buys_computer ="no") = 5/14 = 0.357  
          
     Compute P(X|Ci) for each class:
         
         *P(age = <30 | buys_computer ="yes") = 2/9 = 0.222
         *P(age = <30 | buys_computer ="no") = 3/5 = 0.6
         *P(income ="medium" | buys_computer ="yes") = 4/9 = 0.444
         *P(income ="medium" | buys_computer ="no") = 2/5 = 0.4
         *P(student ="yes" | buys_computer ="yes") = 0.667
         *P(student ="yes" | buys_computer ="no") =  0.2
         *P(credit_rating ="fair" | buys_computer ="yes") = 0.667
         *P(credit_rating ="fair" | buys_computer ="no") = 0.4
         
--> Şimdi mesela X = (age<=30, income = medium, student = yes, credit_rating = fair) için 
--> P(C | X) bulursak bu yeni X örneği için C event'inin gerçekleşme olasılığını bulmuş oluyoruz:
    
--> P(C|X) = P(X|C) P(C) / P(X) 'i hesaplamamız gerek:
    
    * P(X|C) = P(X|C1) * P(X|C2) * ... * P(X|Cn) şeklinde bulanabilir.
      P(X| buys_compuer="yes") =  0.222 * 0.444 * 0.667 * 0.667 = 0.044
      P(X| buys_compuer="no") =  0.6 * 0.4 * ... = 0.019
      
    * P(X|C) P(C) = 0.044 * 0.643 = 0.028
      P(X|C) P(C)' = 0.044 * 0.019 = 0.007
      
    * P(X) iki durum için aynı olduğuna göre bi buys_computer = "yes" ihtimalinin "no" ihtimalinin 4 katı olduğunu söyleyebiliriz.
    
--> Sonuçta yeni gelen datanın yes veya no oluşunu training set'e göre olasılık hesaplayarak buluruz.
          

--> Bu  yöntem lazy learning. Yeni veri gelince olasılıklar ona göre hesaplanıyor.
--> Eager learning için ise gelebilecek her örnek için P(X|Ci) hesaplanır. Daha sonra veri kümesini silebilir.

--> Veri yapısına göre eager learning'in maliyetli olduğu yerler de vardır duruma göre ikisi de kullanılır.


"""


#%%
"""
sklearn kütüphanesini incelersek 4 farklı naive bayes methodu olduğunu görürüz:
    
    --> Gaussian Naive Bayes : Eğer continious bir classication yapmak istersek kullanılır. Sonsuz sınıf yani.
    --> Multinomial Naive Bayes: 0,1,2,3 ... gibi nominal değerleri classify etmeye çalışıyorsak kullanırız.
    --> Bernoulli Naive Bayes: Binary classification için yani binomial değerleri classify etmek için.
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
""" APPLY NAIVE BAYES CLASSIFICATION """

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,Y_train)

y_pred = gnb.predict(X_test)


#Accuracy
print("Test accuracy {}", format(gnb.score(X_test,Y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)
