# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:04:57 2019

@author: Erdo
"""

#%%
""" APRIORI APPLICATION 

--> Şimdiye kadar modeller için sci-kit learn kullanmıştık.
--> Ama scikit learn Apriori modeli içermiyor.

--> Github da bir çok kütüphane bulabiliriz. 
--> Bu derse buna da bakacağız.

--> Eclat'ta aynı şekilde Github'dan bulunabilir.


"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('sepet.csv', header=None) # Bunu yazmasak ilk satırı header olarak alır.
# Dah önceki örneklerde ilk satır header'dı ama burada değil!!!
data.head()

#%%
#apripri() methodu için gerekli parametre formatını oluşturacağız:

t = []
for i in range(0,7501): # her satır için 
    t.append([str(data.values[i,j]) for j in range (0,20)]) # her sütun için elemanları liste içine koyuyoruz.


#%%
from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2) #Methodun içeriğine bakınca transaction'ı list of lists şeklinde : [['A', 'B'], ['B', 'C']] 
#aldığını görüyoruz bizim dataframe'imizi bu formata getirmemiz lazım.
# Bir liste içinde tüm transactionlar tutulacak ve her transtactionda alınan elemanlar yine liste şeklinde tutulacak.

print(list(kurallar))