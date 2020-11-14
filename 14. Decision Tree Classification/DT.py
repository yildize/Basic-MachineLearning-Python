# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:01:16 2019

@author: Erdo
"""
#%%
""" 
DECISION TREE

--> Hatırlarsan, training set'i farklı regionlara ayırıyorduk.
--> Regression için bölgelerin output ortalamalarını alıyorduk yeni geleni ona göre yerleştiriyorduk.

--> Sınıflandırma için diyelim ki boy-kilo için kadın-erkek ayrımı yapıyoruz. Bölgeler homojen olarak ayrılmayabilir.
    * Bu durumda farklı yaklaşımlar olabilir. 
        1) Majority voting: Mesela 3 erkek 2 kadın varsa o region'a kadın diyebiliriz
                * Bunun problemi şu mesela %40 - %60 bölünmüş biz o region'ı direkt erkek kabul ediyoruz.
                * %40'ı gözardı ediyoruz. 

        2) Homojen olana kadar bölmeye devam edebiliriz.
                * Bu durumda da overfitting riski var.
                * Her örneğe kendine has bir region ayrılmaz.
        

"""

"""
Quinlan's ID3 Algorithm'i Naive Bayes için kullandığımız örnek üzerinde inceleyelim:
    
-->Age, Income, Student, Credit_rating featurelarına göre buys_computer yes mi no mu?
-->Bu problemin karar ağacı ile gösterimi şöyle bir şey:
    
    
                                     age?
                                  /   |   \
                                 /    |    \
                                /     |     \
                              <=30   31-40   >40
                             /        |        \
                            /        ALIR       \
                           /                     \
                       Student?               Credit Rating?
                          \                   /        \
                      /     \                /          \
                     /       \              /            \
                    NO       Yes          Excellent     Fair
                    /         \           /               \
                  ALMAZ       ALIR       ALMAZ            ALIR



--> Öğrenme süreci zaten bu ağacı oluşturabilecek, regionları ayırabilmek.
--> Sayısal verilerde bu region'a kesme işlemlerini algoritma kendi yapıyordu. (Bu ayrı bir problem)
--> Nominal veride ise mesela AGE için zaten 3 ihtimal var. 

--> Peki bu karar ağacı nasıl çizildi? Mesela neden age sorusu ile başladık?
--> Neden başka bir feature ile başlamadık?

--> Burada ENTROPİ kavramı devreye giriyor. Detayına bakabilirsin.

--> Temelde entropi hesaplayarak, mesela en üste yaşı alırsak verinin ne kadar iyi dağıldığına bakıyoruz.
--> Eğer veri eşit şekilde dağılırsa bu iyi bir soru demektir. Devam yolu fazla veriyi kısa sürede tüketebiliriz.
--> Bunu matematiksel olarak Information ile hesaplıyoruz.

--> Her feature için hesaplama yapılınca, en fazla information gain yapan feature age bulunur ve en tepeye konulur.
--> Daha sonra mesela < 30 için yine soru sorulabilir o zaman yine information gain hesaplıyoruz.
--> Cevabı netse yanı mesela 31-40 arası hepsi bilgisayar almış daha soru sormaya gerek yok.


--> Bu yönteme ID3 deniyor, başka yöntemlerde var.

"""

#%%
""" DECISION TREE CLASSIFICATION APPLICATION """
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
""" APPLY DECISION TREE """

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion= "entropy")
dtc.fit(X_train,Y_train)

y_pred = dtc.predict(X_test)


#Accuracy
print("Test accuracy {}", format(dtc.score(X_test,Y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)











