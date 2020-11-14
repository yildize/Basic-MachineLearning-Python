# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:38:49 2019

@author: Erdo
"""

"""
Hierarchical Clustering Açıklama Word Dosyasında

-->sklearn clustering'lere google dan bakabilirsin. 
-->Bazı görseller var hangi algoritma farklı dataları nasıl ayırıyor bakmak faydalı olabilir.
"""


#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('musteriler.csv')
data.head()

#%%
X = data.iloc[:,2:4]

#%%
""" Hierarchical Clustering """
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity ='euclidean', linkage='ward')
y_pred = ac.fit_predict(X) # hem fit ediyor hem tahmin ediyor. Burada tahminden kasıt her datapoint için 0,1 veya 2 class'larından hangisine ait olduğunu yazdırmak.


#plot cluster 1
plt.scatter(X[y_pred==0].Hacim, X[y_pred==0].Yas, s=100) 
#y_pred==0 yani 0 kümesine ait olan dataları X'ten çekiyoruz. Yaş'a karşı hacim'i yazdırıyoruz.
#aynı şey aşağıdaki gibi de yapılabilirdi:
#plt.scatter(X.iloc[y_pred==0,0], X.iloc[y_pred==0,1]) 

#plot cluster 2
plt.scatter(X[y_pred==1].Hacim, X[y_pred==1].Yas, s=100) 

#plot cluster 3
plt.scatter(X[y_pred==2].Hacim, X[y_pred==2].Yas, s=100)


#%%
""" ŞİMDİ BAŞKA BİR KÜTÜPHANE KULLANARAK DENDOGRAM KULLANIMINI GÖRECEĞİZ: """

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

#Plot'a göre K=2 veya K=4 mantıklı görünüyor. 3 pek mantıklı görünmüyor!

#%%
""" Hierarchical Clustering """
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity ='euclidean', linkage='ward')
y_pred = ac.fit_predict(X) # hem fit ediyor hem tahmin ediyor. Burada tahminden kasıt her datapoint için 0,1 veya 2 class'larından hangisine ait olduğunu yazdırmak.


#plot cluster 1
plt.scatter(X[y_pred==0].Hacim, X[y_pred==0].Yas, s=100) 
#plot cluster 2
plt.scatter(X[y_pred==1].Hacim, X[y_pred==1].Yas, s=100) 
#plot cluster 3
plt.scatter(X[y_pred==2].Hacim, X[y_pred==2].Yas, s=100)
#plot cluster 4
plt.scatter(X[y_pred==3].Hacim, X[y_pred==3].Yas, s=100)