# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:40:07 2019

@author: Erdo
"""

#%% INTRO
""" 
 * BİRDEN FAZLA REGRESSION VEYA CLASSIFICATION ALG. AYNI ANDA KULLANARAK DAHA BAŞARILI SONUÇLAR ELDE EDİLEBİLİR
 * BU DURUMA ENSEMBLE LEARNING (KOLEKTİF ÖĞRENME) DENİR
 * ADABOOST DA BİR KOLEKTİF ÖĞRENME YÖNTEMİDİR. HATIRLARSAN BİR ÇOK FİLTREYİ BİRLEŞTİRİYORDU.
 
 * RANDOM FOREST TE BİR ENSEMBLE LEARNING METHODUDUR.
 * RANDOM FOREST ASLINDA BİRDEN FAZLA DECISION TREENIN AYNI PROBLEM İÇİN ÇİZİLMESİ VE HEP BİRLİKTE KULLANILMASIDIR.
 
 * MESELA BENİM BİR VERİ SETİM VAR BUNUN FARKLI BÖLGELERİNİ SUBDATASET OLARAK ALIP HER BİRİ İÇİN AYRI BİR DECISION TREE EĞİTEBİLİRİM.
 * DAHA SONRA RANDOM FOREST ALG. BU FARKLI DECISION TREE'LERI MAJORITY VOTING YONTEMIYLE KULLANIP SONUCA ULASIR.
 
 * CLASSIFICATION PROBLEMI İSE, ÇOĞUNLUK HANGI SECENEGI SECTIYSE O OLUR.
 * REGRESSION DA ISE FARKLI DECISION TREES'IN ORTALAMASI ALINIR VE SONUÇ OLARAK KABUL EDİLİR.
 
 * VERİ ARTTIĞI ZAMAN DECISION TREE ALG. İÇİN OVERFITTING'E YOL AÇABİLİR.
 * RANDOM FOREST İLE BU PROBLEM ÇÖZÜLEBİLİYOR AYRICA ÇALIŞMA HIZI DA ARTIRILABİLİR.
"""

#%% APPLICATION

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
""" BUILDING RANDOM FOREST MODEL """

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state=0) # 10 tane decision tree kullanılacak.

rf_regressor.fit(x,y)
y_pred = rf_regressor.predict(x)
s
#plot the final hypothesis
plt.scatter(x,y,color='red')
plt.plot(x,y_pred,color='blue')
plt.show()

# Decision tree daha önceden girilen 10 maaş dışında başka bir değer döndürmüyordu.
# Ancak Random Forest 10 tane D.T sonucunun ortalamasını aldığı için farklı değerler döndürebilir.
# Overfitting olmuyor yani.

# Random Forest tek bir DT'ye göre daha iyi sonuç verir.







