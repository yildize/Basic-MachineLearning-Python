# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:05:45 2019

@author: Erdo
"""

"""" Homework - Tennis Data - Burada Ben Play Tahmin ettim "bad-practice" videoda humidity predict etmiş, mantık aynı! """


#%% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
data = pd.read_csv('odev_tenis.csv') 
data.head()

# %% 
""" Categoric Data to Numeric Data """
# Önce ülke ve cinsiyet sütunlarını, kategorik'ten numeric tipe çevirelim:



#Outlook sütunu
outlook = data.iloc[:,0:1].values #numpy array olarak aldık.
print(outlook)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

outlook[:,0] = le.fit_transform(outlook[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(outlook) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

'''-------------------------'''

# Windy Sütunu:
windy = data.iloc[:,-2:-1].values #numpy array olarak aldık.
print(windy)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

windy[:,0] = le.fit_transform(windy[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(windy) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

windy= ohe.fit_transform(windy).toarray()
print(windy)


# Output Sütunu:
y = data.iloc[:,-1:].values #numpy array olarak aldık.
print(y)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

y[:,0] = le.fit_transform(y[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(y) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

y= ohe.fit_transform(y).toarray()
print(y)

# %% 
""" Building a new Training Set """


outlookColumns = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])  # index'i column'unu kendimiz oluşturuyoruz.
print(outlookColumns)

windyColumns = pd.DataFrame(data=windy, index=range(14), columns=['notWindy', 'windy'])
print(windyColumns)

y = pd.DataFrame(data=y, index=range(14), columns=['notPlay', 'play'])
print(windyColumns)

# dataFrame'leri birleştirme. sexColumns'un sadece 1 sütunu yeterli olacaktır diğeri dummy variable:
dataSet = pd.concat([outlookColumns,data.iloc[:,1:3],windyColumns.iloc[:,1:]], axis=1)
print(dataSet)

# y'yi yaş olarak alalım: 
y = y.iloc[:,-1:]

#%%
""" Feature Scaling: LINEAR REGRESSION ICIN FEATURE SCALING SONUCU HİÇ DEĞİŞTİRMİYOR SANIRIM. """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#dataSet'e feature scaling uygulayalım:
dataSetStd = sc.fit_transform(dataSet) # standardisation

from sklearn import preprocessing 
mms = preprocessing.MinMaxScaler()
dataSetNrm = mms.fit_transform(dataSet) #normalization

# %% 
""" dataSet'in son sütununu y olarak kabul ederek şimdi x_train, y_train, x_set, y_test elde edelim: """
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataSet,y,test_size=0.33, random_state=0) # 2/3 test set, 1/3 test set.

# %% 
""" Multivariate Linear Regression - Building the Model """

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # bir LinearRegression object tanımladık.

regressor.fit(x_train,y_train) # lr objesi training set'i alarak eğitim gerçekleştirir. Cost function'ı minimize eden model parametrelerine ulaşır.


# %% 
""" Application """
y_pred = regressor.predict(x_test) #Eğitilmiş model verilen inputa karşın çıktıları üretir.

# %% 
""" İstenirse Backward Elimination ile Feature Selection Yapılıp Performans Artırılabilir """

 
