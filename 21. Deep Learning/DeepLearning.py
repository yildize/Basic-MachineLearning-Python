# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:35:22 2019

@author: Erdo
"""

#%%
""" 

Bazı DeepLearning kütüphanelerden bahsetmemiz gerekirse:
    
    --> Pytorch
    --> Tensorflow
    --> Caffe
    --> Keras
    --> Deeplearning4J

Tensorflow google tarafından geliştirilen ve python üzerinde çalışan bir kütüphane. 
Diğer dillerde de desteği var.

Tensorflow ile işin en temelinden başlayabiliriz.
NeuralNetworks ile ilgili çok detaylı işleri yapabileceğimiz biz kütüphane.
Tabi NN oluşturmak da daha zahmetli. 

Ancak Keras ile detaylarına çok inmeden çok hızlı şekilde NNs geliştirebiliyoruz.
Keras, tensorflow'un üzerine kurulmuş bir kütüphane gibi düşünülebilir.
Yani tensofrlow ile yapılacak işler yapılıp kütüphaneye çevrilmiş.

Theano ise matematiksel işlemlerin GPU'ya dağıtılmasıyla ilgili bir kütüphane.
"""

#%%
""" Şimdi bir uygulama yapacağız 

--> Amaç bir müşteriyi kaybetmeden önce bunu anlayabilmek.
--> Yeni müşteri edinmek için harcanacak para eski müşteriyi tumak için harcananın 3 misli.

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
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#%%
""" APPLICATION OF NEURAL NETWORKS """

import keras
from keras.models import Sequential # Böylelikle keras'ta bir yapay sinir ağı kullanacağımızı belirttik.
from keras.layers import Dense # Katmanları oluşturabileceğimiz nesne de Dense

classifier = Sequential() #Bu satırla artık RAM'de bir NN tanımladık. Şuan içi boş.

#Artık bu classifier isimli Sequential objesini manipüle ederek istediğimiz özellikte bir NN elde ederiz:

#Bir katman ekleyelim. Dense objesi içinde eklenen katmanın özelliklerini tanımlarız:
classifier.add(Dense(6,init='uniform',activation='relu', input_dim =11))

# Layer'da 6 adet unit olsun dedik. 
# init ile değerlerin 0'a yakın bir şekilde initialize edildiğini söyledik.
# activation function'ı rectifier olarak seçtik.
# input olarak 11 veri aldığını söyledik.

#6 neuron'lu bir hidden layer eklemiştik, şimdi bir hidden layer daha ekleyelim:
classifier.add(Dense(6,init='uniform',activation='relu'))

#İlk layer dışında input_dim belirtmiyoruz, istediğimiz kadar hidden layer ekleyebiliriz.
#Bir kural olmamakla birlikte genelde hidden layer'da act. func. olarak linear functions kullanılır.
#Buna karşın çıkış katmanında sigmoid kullanılır.

#Şimdi çıkış katmanını ekleyelim:
classifier.add(Dense(1,init='uniform',activation='sigmoid'))


#şimdi NN'i compile edeceğiz ve bu süreçte, farklı parametreleri tanımlayacağız.
#mesela hangi optimizasyon methodunu kullanacağız. Stochastic GD mi ne? Biz adam methodu kullanıcaz. Bu da SGD'nin bir versiyonu.
#Bunların detayına keras'ın documentationından bakılabilir.

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])

#cost function'ı da biz tanımlıyoruz. Bunlara da dökümantasyondan bakmakta fayda var.
#Çıkış değerim 0 veya 1 olduğu için binary_crossentropy function'ı kullancağız.
#Kategorik olsa categorical_crossentropy mantıklı olabilirdi.
#optimizasyon sırasında accuracy ve loss yazılacak.

#Artık NN yapısı hazır.

#Şimdi classifier'ımızı fit ededlim:

classifier.fit(X_train,y_train,epochs=50) # Eğitim için verileri 50 tur dönsün dedik.

#CLassifier'ı predict ettirelim:
y_pred = classifier.predict(X_test)
    
#y_pred>0.5 ise 1 kabul edelim diğer türlü 0:
y_pred = (y_pred>0.5)

#%%
""" Bir confusion matrix oluşturalım: """
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


