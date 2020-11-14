# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:11:05 2019

@author: Erdo
"""

"""

VERİ ÖN İŞLEME ŞABLONU

"""


# %% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
missingdf = pd.read_csv('eksikveriler.csv') # Şuan aynı dizinde olduğu için böyle belirtebiliriz. Aynı dizinde değilse absolute path verebiliriz.

# %% 
""" Missing Values """
#Missing value ile başa çıkmak için bir çok farklı yol vardır, bunlardan biri eksik veri yerine ilgili column ortalamasını yazamaktı:.
#Bu iş için sci-kit learn kütüphanesinin altındaki preprocessing alt kütüphanesini kullanalım:
from sklearn.preprocessing import Imputer    
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0) #axis=0 yani sütun bazında mean alacak ve NaN değerleri yerine yazacak.

#Imputer sayısal veriler üzerinde çalışır bu yüzden impute edilecek kısmı ayıralım, yani ülke ve cinsiyet sütunlarını atalım:

NumericPartoftheDataFrame = missingdf.iloc[:, 1:4].values # VALUES alınınca dF yerine Numpy array şeklinde veri alınır.
imputer = imputer.fit(NumericPartoftheDataFrame) # imputer her 3 kolon için ayrı ayrı ortalama değeri hesaplayacak.

#Şimdi imputer kullanarak eksik verileri sütun ortalaması ile dolduralım:
NumericPartoftheDataFrame = imputer.transform(NumericPartoftheDataFrame)
print (NumericPartoftheDataFrame)


# %% 
""" Encoder (Categoric Data --> Numeric Data) """

countryColumn = missingdf.iloc[:,0:1].values #numpy array olarak aldık.
print(countryColumn)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

countryColumn[:,0] = le.fit_transform(countryColumn[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(countryColumn) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

countryColumn = ohe.fit_transform(countryColumn).toarray()
print(countryColumn)

""" TÜM SÜTUNLARI LABEL ENCODİNG ETMENİN KISA YOLU: (ANCAK NUMERİK VERİYİ DE ENCODE EDER)"""  
encodedData = missingdf.apply(LabelEncoder().fit_transform)

# %% 
""" Building Dataframes and Concatanating """

# Şimdi yukarıdaki kısımlarda önce dF'den numeric kısmı ayırım NaN ları doldurduk ve bir numpy elde ettik.
# Daha sonra yien dF'den kategorik ülke sütununu ayırdık ve bunun yerine 3 sütunluk numeric numpy array elde ettik.
# Şimdi bunları birleştirerek yeni bir dataFrame elde edeceğiz:

print(missingdf)
print(NumericPartoftheDataFrame)
print(countryColumn)

# 22x3 ülkeler numpy array'inden bir dataframe elde edelim: 
newDF1 = pd.DataFrame(data=countryColumn, index=range(22), columns=['fr','tr','us'])  # index'i column'unu kendimiz oluşturuyoruz.
print(newDF1)

# 22x3 cleaned numeric values numpy array'inden bir başka dataframe elde edelim:
newDF2 = pd.DataFrame(data=NumericPartoftheDataFrame, index=range(22), columns=['boy','kilo','yas'])
print(newDF2)

# Şimdi cinsiyet verisini alalım:
cinsiyet =  missingdf.iloc[:,-1].values # numpy array olarak cinsiyeti aldık

# df'ye dönüştürelim:
y = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(y)


# dataFrame'leri birleştirme:
dataSet = pd.concat([newDF1,newDF2], axis=1)
print(dataSet)

fulldataSet = pd.concat([dataSet,y], axis=1)
print(fulldataSet)



# %% 
""" Obtain Training and Test Sets """

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataSet,y,test_size=0.33, random_state=0) # 2/3 test set, 1/3 test set.
# Verimiz'in 2/3 ü x_train y_train oldu kalan 1/3 ü x_test y_test oldu!



# %% 
""" Feature Scaling (Standardisation)"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#x_train'e feature scaling uygulayalım:
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)






