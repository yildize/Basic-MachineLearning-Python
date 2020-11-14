# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:29:06 2019

@author: Erdo
"""

#%% INTRO
"""
Suport Vector Machines

    *SVM aslında ilk olarak bir classification alg. olarak ortaya çıkmıç
    *Linearly seperable sınıfları birbirinden, maximum margin ile ayırmaya yarıyordu.
    *İki sınıfı ayıracak birden fazla doğru olabilir SVM ile en büyük margini yaratanı bulabiliriz.
    
"""

#%%
"""
Suport Vector Machines - Regression

    * SVM için olay şuna evriliyor öyle bir doğru buluyoruz ki bir fixed bir margin aralığında maximum data points'i kapsayan doğruyu bulur..
    * Yani aynı kalınlıkta doğrular veya eğriler çizdiğimizi düşünelim, en fazla data point'i kapsayan doğru veya eğriyi seçebiliriz.
    * Kernel function falan kullanarak non-linear SVM hypothesis elde edilebilir.
    
"""

#%% PYTHON APPLICATION

"""
*** SVM KULLANIRKEN MUTLAKA VE MUTLAKA FEATURE SCALING YAPMALIYIZ, SVM'IN OUTLIER VERİLERE KARŞI HASSASİYETİ VAR.
"""

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
plt.scatter(x,y,color='red')
plt.scatter(x_scaled,y_scaled,color='green')


# %% 
""" Building the SVR Model  - Hem x hem y scaled"""

from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf") # Başka kernellerde kullanılabilir, aşağıdaki linklere bakılabilir:

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

svr_regressor.fit(x_scaled,y_scaled)
y_pred = svr_regressor.predict(x_scaled)

#plot the final hypothesis
plt.scatter(x_scaled,y_scaled,color='red')
plt.plot(x_scaled,y_pred,color='blue')

