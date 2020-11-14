# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:59:44 2019

@author: Erdo
"""

#%%
""" Algoritmaların Başarısını Nasıl Ölçeriz? 

 --> İlk yöntem R2 (R-Square) Yöntemi (1 en iyi - 0 çok kötü - Negatif Çöp)
 
     * Hata Karelerinin Toplamı =  Sum [(hi - yi)^2] tüm veriler boyunca.
     * Ortalama Farkların Toplamı = Sum [(hi - yavg)^2] tüm veriler boyunca.
     
     * R2 =  1 - (HKT / OFT)
     
     * BOY   KİLO  YAŞ   TAHMİN 
     
       180    74   24     25
       170    75   30     28
       192    85   27     30

                  Avg=27
                  
    * Yukarıdaki gibi bir örnekte: 
        
        HKT = 1^2 + 2^2 + 3^2 şeklinde hesaplanırken
        OFT = 2^2 + 1^2 + 3^2 şeklinde hesaplanabilir.
        
        
    *Sonuçta OFT bize aslında olabilecek en basit ve temel tahmin methodunun sonucunu sunuyor.
    *Yaşların ortalamasını alıp tahmin ettirdiğimizi varsayarsak. R2 = 0 çıkar bu en kötü sonuç.
    *Eğer benim algoritmam için R2 < 0 çıkıyorsa algoritmam leş demektir çöpe at.
    
    *Bunun dışında eğer tüm örnekleri doğru tahmin edersek HKT = 0 dır bu da R2 = 1 demek olur yani ideal durum. 

"""

#%%
""" 2. YÖNTEM ADJUSTED R2 METHODU 

 --> R2 yönteminin bazı sıkıntıları var: (Yeni eklenen değişken R2 değerini asla düşürmüyor!!! Olumsuz etkiyi göremiyoruz.) 
     
     *Ben evaluation için R2 kullanıyorsam, amacım R2 = 1 verecek modeli elde etmektir.
     *Modelde bir değişiklik yapınca eğer R2 artıyorsa bu değişiklik iyidir, azalıyorsa kötüdür diyebiliriz.
     *Ancak R2 bazı durumlarda bu durumları tam olarak evaluate edemiyor.
     
     *Mesela bir multivariate linear regression problemimiz olsun.
     *Çalışanların belirli feature'larına göre maaş tahmini yaptırıyorum ve ona göre ödeme yapıyorum.
     
     *İlk modelim  h = Theta0 + Theta1*x1 
     *Yeni bir model denemek istedim bir feature daha ekledim:  h = Theta0 + Theta1*x1 + Theta2*x2
     *Eğer x2 işe yarar bir feature ise R2 değerim artacaktır.
     *FAKAT x2 useless ise eğitim sonunca Theta2 0 değerine yaklaşaktır ve x2'nin etkisi silinecektir.
     *BÖYLECE R2 sonucu olduğu yerde kalacak yani, x2'nun kötü bir feature olduğunu söyleyemeyecektir.
     
     *Bu bir problem bunu çözmek için ADJUSTED R2 kullanıyor:
         
--> Adjusted R2 = 1 - (1-R2) * (n-1) / (n-p-1)
 
     * Bu methodun da problemleri var tabi ama en azından bu problem özelinde R2'e bir alternatif sağlar.
     
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
data = pd.read_csv('maaslar.csv') 
data.head()

# %% 
""" Obtain the training set x and y """

x = data.iloc[:,1:2] #eğitim seviyesini numeric olarak aldık.
y = data.iloc[:,2:] #maasları aldık.

# %% 
""" Feature Scaling - SVR İLE HEM X HEM Y HER ZAMAN SCLAE EDİLMELİ, SAYILAR YAKIN OLMAZSA PROBLEM ÇIKARIYOR """

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_scaled = sc.fit_transform(x)
y_scaled = sc.fit_transform(y)
# %% 
""" Training set'i görselleştirelim """
plt.figure()
plt.scatter(x,y,color='red')
plt.title('Original Data')
plt.show()

plt.figure()
plt.scatter(x_scaled,y_scaled,color='green')
plt.title('Scaled Data')
plt.show()

# %% 
""" Linear Regression Building the Model """

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor.fit(x,y)
y_pred_LR = lin_regressor.predict(x)

#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred_LR,color='blue')
plt.title('Linear Regression Model on Original Data')

# %% 
"""Polynomial Regression Building the Model (DİKKAT ET DAHA ÖNCE LR İÇİN XO=1 EKLEMEMİŞTİK FAKAT BURADA EKLEDİK DEMEKKİ EKLENİYOR) """

from sklearn.preprocessing import PolynomialFeatures
polyFeatureModel = PolynomialFeatures(degree=4) # x0, x1, x1^2  şeklinde featureları oluşturacak modeli elde ettik. 
polyFeatures = polyFeatureModel.fit_transform(x) # polyFeatures içinde artık ilgili featureları tutar.

# BU NOKTADAN SONRA ARTIK POLY FEATURES numpy array olarak HAZIR OLDUĞUNA GÖRE LİNEAR REGRESSOR İLE POLY MODEL KURULABILIR.
lin_regressor2 = LinearRegression() # bir LinearRegression object tanımladık.

lin_regressor2.fit(polyFeatures,y) #polynomial regressor gibi düşünülebilir.
y_pred_PR = lin_regressor2.predict(polyFeatures) #polynomial regressor'ın tahminleri

#plot the polynomial final hypothesis
plt.scatter(x,y,color='red') #normal data pointler
plt.plot(x,y_pred_PR,color='blue') #polynomial regressor'ın tahminleri.
plt.title('Polynomial Regression Model on Original Data')
# %% 
""" Building the SVR Model  - Hem x hem y scaled"""

from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf") # Başka kernellerde kullanılabilir, aşağıdaki linklere bakılabilir:

svr_regressor.fit(x_scaled,y_scaled)
y_pred_SVR = svr_regressor.predict(x_scaled)

#plot the final hypothesis
plt.scatter(x_scaled,y_scaled,color='red')
plt.plot(x_scaled,y_pred_SVR,color='blue')
plt.title('Support Vector Regression Model on Scaled Data')
# %% 
""" Building the Decision Tree Model """

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)


dt_regressor.fit(x,y)
y_pred_DT = dt_regressor.predict(x)

#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred_DT,color='blue')
plt.title('Decision Tree Model on Original Data')

# %% 
""" Building the Random Forest Model """

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state=0) # 10 tane decision tree kullanılacak.

rf_regressor.fit(x,y)
y_pred_RF = rf_regressor.predict(x)

#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred_RF,color='blue')
plt.title('Random Forest Model on Original Data')


#%%
""" R2 EVALUATION """

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







