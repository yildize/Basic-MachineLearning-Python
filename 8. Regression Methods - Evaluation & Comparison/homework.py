# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:31:11 2019

@author: Erdo
"""


""" HOMEWORK

 Feature selection yap
 5 farklı algoritma ile model oluştur
 Yöntemleri karşılaştır
 10 yıl tecrübeli 100 puanlık bir CEO ile aynı özellikli bir Müdürün maaşlarını tahmin ettir. Sonuçları yorumla

 """

"""  REGRESSION ALGORİTMALARI ADV. - DISADV.

     Linear Reg.
         + Veri boyutundan bağımsız olarak doğrusal ilişki üzerine kuruludur.
         - Doğrusallık kabulu aynı zamanda hatadır.
         
     Polynomial Reg.
         +Doğrusal olmayan problemleri adresler
         -Başarı için doğru polinom derecesi önemlidir.
    
     SVR
         +Doğrusal olmayan modellerde çalışır, marjinal değerlere karşı ölçekleme ile dayanıklı olur.
         -Ölçekleme önemlidir.Doğru kernel seçimi önemlidir.
         
    Decision Tree
        +Ölçeklemeye gerek yok. Doğrusal veya doğrusal olmayan problemlerle çalışır.
        -Sonuçlar sabitlenmiştir, küçük veri kümelerinde overfitting ihtimali yüksektir.
    
    Random Forest
        +Ölçeklemeye ihtiyaç duymaz. Lin or Nonlin prob. için çalışır. Overfitting riski düşüktür.
        -Çıktıların yorumu ve görselleştirilmesi görece zordur.
        



"""

#%%
""" PYTHON APPLICATION MODELS """

#%% 
""" Library Import """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% 
""" Data Import """
data = pd.read_csv('maaslar_yeni.csv') 
data.head()

# %% 
""" Building Dataframes and Concatanating """

# Şimdi maaş verisini alalım:
y =  data.iloc[:,-1:] # numpy array olarak cinsiyeti aldık
print(y)

# Features olarak UNVAN - KIDEM - PUAN alındı:
x = data.iloc[:,2:5]
print(x)


# %% 
""" Feature Scaling - SVR İLE HEM X HEM Y HER ZAMAN SCLAE EDİLMELİ, SAYILAR YAKIN OLMAZSA PROBLEM ÇIKARIYOR """

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_scaled = sc.fit_transform(x)
y_scaled = sc.fit_transform(y)

# %% 
""" Linear Regression Building the Model """

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor.fit(x,y)
y_pred_LR = lin_regressor.predict(x)

# %% 
"""Polynomial Regression Building the Model (DİKKAT ET DAHA ÖNCE LR İÇİN XO=1 EKLEMEMİŞTİK FAKAT BURADA EKLEDİK DEMEKKİ EKLENİYOR) """

from sklearn.preprocessing import PolynomialFeatures
polyFeatureModel = PolynomialFeatures(degree=4) # x0, x1, x1^2  şeklinde featureları oluşturacak modeli elde ettik. 
polyFeatures = polyFeatureModel.fit_transform(x) # polyFeatures içinde artık ilgili featureları tutar.

# BU NOKTADAN SONRA ARTIK POLY FEATURES numpy array olarak HAZIR OLDUĞUNA GÖRE LİNEAR REGRESSOR İLE POLY MODEL KURULABILIR.
lin_regressor2 = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor2.fit(polyFeatures,y) #polynomial regressor gibi düşünülebilir.
y_pred_PR = lin_regressor2.predict(polyFeatures) #polynomial regressor'ın tahminleri

# %% 
""" Building the SVR Model  - Hem x hem y scaled"""

from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf") # Başka kernellerde kullanılabilir, aşağıdaki linklere bakılabilir:

svr_regressor.fit(x_scaled,y_scaled)
y_pred_SVR = svr_regressor.predict(x_scaled)

# %% 
""" Building the Decision Tree Model """

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)


dt_regressor.fit(x,y)
y_pred_DT = dt_regressor.predict(x)

# %% 
""" Building the Random Forest Model """

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state=0) # 10 tane decision tree kullanılacak.

rf_regressor.fit(x,y)
y_pred_RF = rf_regressor.predict(x)


#%%
""" R2 EVALUATION - OLS SONUCUNDA ÇIKAN R2 İLE BUNLAR FARKLI OLUR """

from sklearn.metrics import r2_score

#R2 SCORE FOR LINEAR REGRESSION
print(r2_score(y, y_pred_LR))

#R2 SCORE FOR POLYNOMIAL REGRESSION
print(r2_score(y, y_pred_PR))

#R2 SCORE FOR SUPPORT VECTOR MACHINE
print(r2_score(y_scaled, y_pred_SVR))

#R2 SCORE FOR DECISION TREE
print(r2_score(y, y_pred_DT))

#R2 SCORE FOR RANDOM FOREST
print(r2_score(y, y_pred_RF))
#%% 
""" PRINT OLS SUMMARY FOR FEATURE SELECTION"""
import statsmodels.api as sm

x_new = sm.add_constant(x) # Başına 1 column'u ekledik.
model = sm.OLS(y,x_new) 
results = model.fit()
print(results.summary())
#print(results.params)

#%%
""" OLS SONUÇLARINA GÖRE 2. VE 3. FEATURES İÇİN P-VALUE > 0.05 BUNLARI ATALIM:"""
xx = data.iloc[:,2:3]
xx_scaled = sc.fit_transform(xx)


#%% 
""" PRINT OLS SUMMARY FOR FEATURE SELECTION"""
import statsmodels.api as sm

xx_new = sm.add_constant(xx) # Başına 1 column'u ekledik.
model = sm.OLS(y,xx_new) 
results = model.fit()
print(results.params)
print(results.summary())

# GÖRÜYORUZ Kİ R2 VALUE DÜŞTÜ O ZAMAN ATMAMAK DAHA MANTIKLI

#%% 
""" Yine de Yeni DATASET ile modelleri tekrar oluşturalım: """

""" Linear Regression Building the Model """

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor.fit(xx,y)
y_pred_LR = lin_regressor.predict(xx)

"""Polynomial Regression Building the Model (DİKKAT ET DAHA ÖNCE LR İÇİN XO=1 EKLEMEMİŞTİK FAKAT BURADA EKLEDİK DEMEKKİ EKLENİYOR) """

from sklearn.preprocessing import PolynomialFeatures
polyFeatureModel = PolynomialFeatures(degree=4) # x0, x1, x1^2  şeklinde featureları oluşturacak modeli elde ettik. 
polyFeatures = polyFeatureModel.fit_transform(xx) # polyFeatures içinde artık ilgili featureları tutar.

# BU NOKTADAN SONRA ARTIK POLY FEATURES numpy array olarak HAZIR OLDUĞUNA GÖRE LİNEAR REGRESSOR İLE POLY MODEL KURULABILIR.
lin_regressor2 = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor2.fit(polyFeatures,y) #polynomial regressor gibi düşünülebilir.
y_pred_PR = lin_regressor2.predict(polyFeatures) #polynomial regressor'ın tahminleri

""" Building the SVR Model  - Hem x hem y scaled"""

from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf") # Başka kernellerde kullanılabilir, aşağıdaki linklere bakılabilir:

svr_regressor.fit(xx_scaled,y_scaled)
y_pred_SVR = svr_regressor.predict(xx_scaled)

""" Building the Decision Tree Model """

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)


dt_regressor.fit(xx,y)
y_pred_DT = dt_regressor.predict(xx)

""" Building the Random Forest Model """

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state=0) # 10 tane decision tree kullanılacak.

rf_regressor.fit(xx,y)
y_pred_RF = rf_regressor.predict(xx)


#%% 
""" Tekrar R2 Evaluation - Tek Feature İçin"""
from sklearn.metrics import r2_score

#R2 SCORE FOR LINEAR REGRESSION
print(r2_score(y, y_pred_LR))

#R2 SCORE FOR POLYNOMIAL REGRESSION
print(r2_score(y, y_pred_PR))

#R2 SCORE FOR SUPPORT VECTOR MACHINE
print(r2_score(y_scaled, y_pred_SVR))

#R2 SCORE FOR DECISION TREE
print(r2_score(y, y_pred_DT))

#R2 SCORE FOR RANDOM FOREST
print(r2_score(y, y_pred_RF))



#%% 
""" Feature Arası İlişkiyi Görmek İçin Alternatif Bir Yöntem Var:"""
import seaborn as sns  # visualization tool

x.corr()

#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

