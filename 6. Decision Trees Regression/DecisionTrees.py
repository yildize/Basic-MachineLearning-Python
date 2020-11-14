# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 01:26:00 2019

@author: Erdo
"""

#%% INTRO
"""
Decision Trees

    *Decision Tree algoritması da ilk olarak bir classification algoritması olarak çıkmıştır.
    *Genlde classification için kullanılsa da regression için de kullanılabiliyor.
    
"""

#%% 
"""
Diyelim ki BOY ve KİLO feature'larımız var ve YAŞ'ı tahmin etmek istiyorum:

    *DATA'yı 2 boyutlu BOY-KİLO uzayda gösterebiliriz. 
    *Decision tree bu 2 boyutlu uzayı önce ikiye ayırıyor. Bu ayırım entropi ile ilgili şuan bundan bahsetmeyeceğiz
    *Bu bölme işlemleri zaten eğitim sürecini oluşturuyor.
    *Datayı sürekli böleceğiz ve gruplandıracağız böylece, yeni gelen data o gruplardan hangisine düşüyorsa ona göre tahmin yapacağız.
    *Aslında bu classification oluyor.
        
                                   |Boy > 165| ...
                                 /
                     |Kilo > 75|
                   /             \ |Boy < 165|
         |Boy 145|               
                   \ |Kilo < 75| ..
                   
                   
    *Data set yaprak düğüm sayısı kadar bölgeye ayrılır. 
    *Her bölge için YAŞ ortalaması bulunur ve bu ortalama artık o bölgenin yaş değeri olarak atanır.
    
    *Sonuçta yeni data gelince BOY - KİLO bilgisine göre bu tree'de ilgili yaprağa yerleşir ve classification yapılmış olur.
    *Böylece o region için ilgili YAŞ değeri tahmin edilir.
    
"""


##% PYTHON APPLICATION
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
""" Training set'i görselleştirelim """
plt.scatter(x,y,color='red')
plt.show()


# %% 
""" Building the Decision Tree Model"""

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)


dt_regressor.fit(x,y)
y_pred = dt_regressor.predict(x)

#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred,color='blue')

# BURADA DecisionTree modeli her noktayı ezberledi, yani 10 farklı bölge oluşturdu ve 10 output hazırladı.
# Bu 10 bölgenin içine düşen her input için aynı sonuçları verecektir.



