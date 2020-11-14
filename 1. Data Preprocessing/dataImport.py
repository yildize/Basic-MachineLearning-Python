# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:18:08 2019

@author: Erdo
"""


# %% Library Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% DataImport
data = pd.read_csv('veriler.csv') # Şuan aynı dizinde olduğu için böyle belirtebiliriz. Aynı dizinde değilse absolute path verebiliriz.

# %% Pandas
boy =  data[['boy']]  # Çift parantez yapmazsak type(boy) series olur ama şuan dataFrame
print(boy)

boykilo = data[['boy','kilo']]
print(boykilo)



# %% MISSING VALUES
missingdf = pd.read_csv('eksikveriler.csv')

#Missing value ile başa çıkmak için bir çok farklı yol vardır, bunlardan biri eksik veri yerine ilgili column ortalamasını yazamaktı:.
    #Bu iş için sci-kit learn kütüphanesinin altındaki preprocessing alt kütüphanesini kullanalım:
from sklearn.preprocessing import Imputer
    
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0) #axis=0 yani sütun bazında mean alacak ve NaN değerleri yerine yazacak.

#Imputer sayısal veriler üzerinde çalışır bu yüzden impute edilecek kısmı ayıralım, yani ülke ve cinsiyet sütunlarını atalım:
NumericPartoftheDataFrame = missingdf.iloc[:, 1:4].values # VALUES alınınca dF yerine Numpy array şeklinde veri alınır.
print(NumericPartoftheDataFrame)

imputer = imputer.fit(NumericPartoftheDataFrame) # imputer her 3 kolon için ayrı ayrı ortalama değeri hesaplayacak.

#Şimdi imputer kullanarak eksik verileri sütun ortalaması ile dolduralım:
NumericPartoftheDataFrame = imputer.transform(NumericPartoftheDataFrame)
print (NumericPartoftheDataFrame)


#%%
"""
 VERİ GRUPLARI
 
 --> Verileri 2 ana grupta inceleyebiliriz:
     
       1) CategoriC Veriler: Mesela bir kişinin sigare içip içmemesi. Cinsiyeti. Eğitim durumu. İllerin pLaka numarası, vb. gibi.
       Bunları Evet, Hayır, İlkokul, Lise, Lisans, Master, PhD gibi değerler.
       
                1.a Nominal: 
                              -->Ne sıralanabilir ne ölçülebilir. Mesela plaka numaraları.
                              -->Mesela araba markaları. Bunları sıralamayız da.
                             
                1.b Ordinal: 
                             --> Order'dan gelir. Sıraya sokulabilen aralarında büyük küçük ilişkisi olan ama ölçülemeyen veriler.
                             --> Plaka numaraları sıralanabilir ama İstanbul (34) > Antalya (07) nin bi anlamı yok.
       
       2) Numeric Veriler:
    
                2.a Oransal(ratio)
                            --> Birbiri ile çarpılıp bölünebilen veriler.
                             
                2.b Aralık (interval)
                            --> Çarpma bölme gibi işlemleri kabul etmez.
                            
"""

#%%
"""
KATEGORİK VERİLERİ SAYISAL VERİLERE ÇEVİRMEK

     -->Biz ML uygulamarı için Kategorik verileri de kullanmak zorunda olabiliriz. 
     -->Ancak bunun için kategorik verileri numerik verilere çevirmeliyiz.
     
     -->Mesela ülke sütununu ben sayısal veriye çevirmek istiyorum. Doğrudan 0-Tr, 1-USA, 2-England olsun diyebiliriz.
     -->Ama bu biraz riskli. Sadece 2 opsiyon varsa 0 ve 1 ile çevirebiliriz ama daha çok ise bu biraz sıkıntı.
     
     -->Bunun yerine Mesela 3 ülke opsiyonu varsa her bir example için 3 yeni feature eklerim.
     -->Tr olanların Tr feature'ı 1 USA ve England feature'ı 0 olur.
     
     --> Ülke  Boy  Kilo  Yaş       Böyle bir data set yerine artık aşağıdaki gibi bir data set elde ederiz:
         Tr    180  73    24
         
     --> Tr USA England Boy Kilo Yaş
         1   0    0     180  73  24
         

"""

#%% KATEGORİK VERİYİ SAYISAL VERİYE ÇEVİRMEK İÇİN KOD
dframe = pd.read_csv('veriler.csv')
countryColumn = dframe.iloc[:,0:1].values #numpy array olarak aldık.
print(countryColumn)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

countryColumn[:,0] = le.fit_transform(countryColumn[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(countryColumn) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')
countryColumn = ohe.fit_transform(countryColumn).toarray()
print(countryColumn)


#%% VERİLERİN BİRLEŞTİRİLMESİ VE DATAFRAME OLUŞTURULMASI

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




#%% VERİYİ TRAINING AND TEST SET OLARAK BÖLMEK

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataSet,y,test_size=0.33, random_state=0) # 2/3 test set, 1/3 test set.
# Verimiz'in 2/3 ü x_train y_train oldu kalan 1/3 ü x_test y_test oldu!


#%% 
"""
FEATURE SCALING (ÖZNİTELİK ÖLÇEKLEME)

 -->Verilerin farklı eksen limitlerinde olması yani farklı dünyalardan olması veri biliminde problem teşkil eder.
 -->Bizim bu verileri ortak bir dünyada birleştiririz. Bunun için 2 farklı yol var:
     
      1. Standardisation: z = (x-Mu) / Sigma  where Mu: Mean, Sigma: Standard deviation. --> Ortalama değer 0 olur diğerleri - veya + olabilir.
      
      2. Normalization: (x-min(x))/[max(x)-min(x)] --> Veri 0-1 arasına oturur. Min 0 olur Max 1 olur.
      
      
Standartlaştırmanın problemi OUTLIERs yani eğer yanlışlıkla çok büyük veya çok küçük bir değer girilirse
bu standartlaştırma sonucunda tüm datayı çok bozar.

Ancak Normalizasyon bu durumdan daha az etkilenir.

    
"""

#%% Feature Scaling Python

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#x_train'e feature scaling uygulayalım:
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)






