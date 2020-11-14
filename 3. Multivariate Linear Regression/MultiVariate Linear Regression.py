# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:53:32 2019

@author: Erdo
"""


'''
GİRİŞ 
-----

--> Bir önceki örnekte tek feature vardı. Şimdi birden fazla feature olan bir durumu inceleyeceğiz:
--> Mesela elimizde insanların ülke, boy, kilo, cinsiyet ve yaş bilgileri var.
--> Bir multivarate linear regression model eğiterek yeni insanların yaşını; ülke,boy,kilo,cinsiyet bilgisine göre tahmin etmeye çalışacağız.
'''

'''
Multivariate Linear Regression 
-----

--> Simple LR hipotezi: h = ax+b formunda iken. MLR'da bu h=theta0*x0+ theta1*x1 + theta2*x2 + ... şeklinde gidiyor.
--> Mesela elimizde insanların ülke, boy, kilo, cinsiyet ve yaş bilgileri var.
--> Bir multivarate linear regression model eğiterek yeni insanların yaşını; ülke,boy,kilo,cinsiyet bilgisine göre tahmin etmeye çalışacağız.
'''

'''
Dummy Variable
-----

--> Eğer bir feature'ı tek bir sütun ile anlatabilirken birden fazla sütun kullanıyorsak bu sıkıntı.
--> Her feature'ın modele etkisi aynı olmalı ama eğer bir feature'ı 5 kolonda yazarsak bu resultant hypothesis'i bozar.
--> Hangi ML algoritma olduğuna da bağlı bazıları buna dayanıklıdır. 
--> Bağışıklı olmayanlar için bir feature ne kadar çok tekrar edilirse sonuç giderek o feature'dan daha çok etkilenir.

--> SANIRIM BU DURUM CLASSIFICATION İÇİN DAHA AZ PROBLEM TEŞKİL EDİYOR AMA REGRESSION İÇİN DAHA ÖNEMLİ.

'''

'''
 p - value (probability value)

--> H0: Null Hypothesis : Bir kabulu ifade eder. Mesela üretim yapıyoruz her kutuda 100 kurabiye vardır hipotezi.
--> H1: Alternative Hypothesis : H0'ın tersi, her kutuda 100 kurabiye yoktur.

--> p-value : bize H0'ın güvenilirliği ile ilgili bir bilgi veriyor.
--> P-value küçüldükçe H0'ın hatalı olma ihtimali artar.
--> Significance Level bir treshold'tur ve p-value < SL ise H0 hatalıdır deriz bu değer genelde 0.05 alınır.

--> Biz p-value'yu feature selection için kullanabiliriz.

'''

'''
    FEATURE SELECTION

--> Multivariate linear regression için nasıl feature selection yapmalıyız?
--> Tüm değişkenleri seçmek her zaman en iyi çözüm değildir. Bazen sonucu kötü etkileyebilir.

--> Feature selection için 5 farklı yoldan bahsedebiliriz:
    
    1. Bütün Değişkenleri Dahil Etmek (Bu yöntem bazen işe yarar veya zorunluluktan seçiyor olabiliriz.)
    
    2. Backward Elimination
    
       *Önce bir Significance Level (SL) seçilir bunu başarı kriteri gibi düşünebiliriz. Genelde SL =  0.05 olarak seçilir.
       *Sonra tüm features kullanılarak bir model inşa edilir.
       *p-values hesaplanır ve en yüksek p value'su olan  feature ele alınır eğer P>SL ise 4. adım değilse 6. adıma gidilir.
       *P>SL olan feature sistemden kaldırılır.
       *Kalan değişkenlerle yeni model kurulur ve 3. adıma geri dönülür.
       *Makine öğrenmesi sonlandırılır.
     
    3. Forward Elimination: Backward'a benzer, her adımda tüm featurelar arasından en iyisini seçip sisteme ekliyoruz.
        
        *SL=0.05 seçilir
        *Tüm features kullanılarak bir model inşa edilir.
        *En düşük p-value değerine sahip olan değişken ele alınır.
        *3. adımdaki değişken sabit tutularak yeni bir değişken daha seçilir ve sisteme eklenir.
        *Makine öğrenmesi güncellenir ve 3. adıma dönülür. En düşük p-value olan değişken için p<SL sağlanıyorsa 3. adıma dön. Değilse devam
        *Sonlandır.
        
    4. Bidirectional Elimination: Forward ve Backward'ın birleşimi.
    
    5. Score Comparison
    
        * Modeli evaluate etmek için bir başarı kriteri belirleriz.
        * Daha sonra bütün olası modelleri veya feature selections için modelleri kurarız 
        * Hepsini evaluate ederiz ve en iyi performans göstereni seçeriz.
        
'''




#%% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
data = pd.read_csv('veriler.csv') 
data.head()

# %% 
""" Categoric Data to Numeric Data """
# Önce ülke ve cinsiyet sütunlarını, kategorik'ten numeric tipe çevirelim:

# Ülke Sütunu:
country = data.iloc[:,0:1].values #numpy array olarak aldık.
print(country)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

country[:,0] = le.fit_transform(country[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(country) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

country = ohe.fit_transform(country).toarray()
print(country)

'''-------------------------'''

# Cinsiyet Sütunu:
sex = data.iloc[:,-1:].values #numpy array olarak aldık.
print(sex)

from sklearn.preprocessing import LabelEncoder # label encoderi kategorik değerleri numaralandırması için kullanırız.
le = LabelEncoder()

sex[:,0] = le.fit_transform(sex[:,0]) # artık tr yerine 1 us yerine 2 fr yerine 0  yazıldı.
print(sex) # fakat hala veriler tek bir sütun halinde. Bunu yukarıdak açıklanan satır formatına çevirelim:

from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(categorical_features='all')

sex = ohe.fit_transform(sex).toarray()
print(sex)


# %% 
""" Building a new Training Set """

# 22x3 ülkeler numpy array'inden bir dataframe elde edelim: 
countryColumns = pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])  # index'i column'unu kendimiz oluşturuyoruz.
print(countryColumns)

# 22x3 cleaned numeric values numpy array'inden bir başka dataframe elde edelim:
sexColumns = pd.DataFrame(data=sex, index=range(22), columns=['e', 'k'])
print(sexColumns)


# dataFrame'leri birleştirme. sexColumns'un sadece 1 sütunu yeterli olacaktır diğeri dummy variable:
dataSet = pd.concat([countryColumns,data.iloc[:,2:4],sexColumns.iloc[:,0:1]], axis=1)
print(dataSet)

# y'yi yaş olarak alalım: 
y = data.iloc[:,1:2]

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
""" Backward Elimination ile Feature Selection """
import statsmodels.formula.api as sm

#Öncelikle dataSet'in ilk sütununa 1'leri ekleyelim.
#Bunu model oluştururken kendi ekliyordu sanırım o yüzden biz eklemedik.
X = np.append(arr = np.ones((22,1)).astype(int), values = dataSet, axis=1)  #dataSet as a numpy array.

X_list = dataSet.iloc[:,[0,1,2,3,4,5]].values #İlk olarak tüm features alındı ve her biri için p-values hesaplanacak:
r_ols= sm.OLS(endog = y, exog = X_list).fit() # X_list içindeki her feature'ın y üzerindeki etkisini ölçer ve ona göre p-values döndürür.
print(r_ols.summary()) #r_ols bize bir rapor döndürür. Bu raporda bir çok values yanında p-values de vardır.

# En yüksek p değerine sahip olan x5'tir x5 için p=0.717 > SL 0.05 olduğu için x5 i eleyip aynı process'i tekrar ederiz:
X_list = dataSet.iloc[:,[0,1,2,3,5]].values 
r_ols= sm.OLS(endog = y, exog = X_list).fit() 
print(r_ols.summary()) 

#Baktık bu sefer de x5 sıkıntı, çok büyük değil ama istersek eleyebiliriz:
X_list = dataSet.iloc[:,[0,1,2,3]].values 
r_ols= sm.OLS(endog = y, exog = X_list).fit() 
print(r_ols.summary()) 


# Sıkıntı çözüldü demekki model için sadece 0,1,2,3. değişkenleri kullanmak daha iyi deriz.
# Ancak burada başka parametreler de önemli sadece buna güvenmek pek doğru değil. 
# İleride daha detaylı göreceğiz.

""" BENCE BURADA HATA VAR X_list oluşturulurken X'i kullanmalıydık dataSet'i değil!."""
""" ANLADIĞIM KADARIYLA OLS.fit() ile bir regression model eğitiliyor
    Daha sonra summary ile bu eğitilen modelin R valuesuna falan bakılabiliyor.
    
   x = sm.add_constant(x) ile x dataset'inin başına 1 column'u ekleyebiliriz.
    
"""



