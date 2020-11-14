# -*- coding: utf-8 -*-
"""
Spyder Editor

Author : Erdo
"""

#%%
"""
--> Unsupervised Learning için: K-means Clustering ve Hierarchical Clustering göreceğiz.
--> Clustering'in başarı kriteri class'ın içindeki örneklerin kendi arasında mesafesini min tutarken,
--> Class'ların birbiri arasındaki mesafeyi max tutmaktır.

--> Clustering nerelerde kullanılır:
    
    *Customer Segmentation
        - Collabration filtering
        - Özel kampanyalar
        - Fraud detection : Class'ların dışında kalan örnekler
        - Eksik verilerin tamamlanması
        - Verilerin alt kümesi üzerinde yapılan bütün işlemler

    *Market Segmentation
        - Davranışsal segmentasyon
        - Demografik segmentasyon
        - Psikolojik segmentasyon
        - Coğrafi segmentasyon
        - Verinin alt kümesi üzerinde yapılan bütün işlemler
        
    *Computer Vision
        -Mesela sağlıkta, görüntülerden farklı örüntüler yakalaması işe yarar.
        -Genel olarak nesne tanıma vb gibi yerlerde, hareket yakalama da vb. clustering kullanılır.
        -Farklı objeleri ayırt etmeye yarıyor.
            
"""

#%%
""" K-MEANS CLUSTERING

--> Küme sayısı K'yı kullanıcıdan alır.
--> Rastgele olarsk K merkez seçilir.
--> Her data point kendine en yakın merkeze atanır.
--> Her küme için yeni merkez noktası hesaplanır ve merkez kaydırılır.
--> Bu şekilde adım adım devam eder.

--> Klasik K-means için ilk merkezler veriler arasından rastgele seçilir.
--> Ama alternatifleri de var mesela en uzak noktaları almak gibi.

"""


#%%
""" K-MEANS RASSAL BAŞLANGIÇ TUZAĞI

--> K-means'in bazı problemleri var
--> Andrew'in de bahsettiği gibi bazen yanlış initial centeroid ataması, bazen local optima da sıkışmayı getirebilir.
--> Bu sebeple çok kez random initialization yapıp min J() vereni seçiyorduk.

--> Bu local optima'ya takılmaktan kurtulmak için farklı yöntemler geliştirilmiş mesela K-means++. 

"""



#%%
""" K-MEANS KÜME SAYISI K'YI SEÇMEK 

--> Farklı K sayılarını deneyip sonuçları evaluate edebiliriz.
--> Andrew bir cost function tanımlamıştı.

--> Soru bu cost function'ın nasıl tanımlanacağı, farklı yaklaşımlar var sanıyorum.
--> WCSS: Her bir cluster için o cluster'daki elemanların merkezlerine olan mesafelerinin toplamı.
--> Bu andrew'in dediği ile aynı sanıyorum.

--> Elbow metodundan bahsediyor. Farklı K'lar için J() çizeriz buradan optima bulunur.

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
""" K-Means Application """
from sklearn.cluster import KMeans

kmeans =  KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

#%%
""" Farklı K sayıları için J bulalım ve çizdirelim """
results =[] #
for i in range(1,10):
    kmeans = KMeans (n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X)
    results.append(kmeans.inertia_)
    
    
plt.plot(results)

# 2 3 veya 4 elbow olarak alınabilir.

# UNUTMA J küçük demek en iyi demek değil, overfitting demek anlamına gelebilir.
# Burada elbow'a bakılır 
# Veya bölündükten sonra data kullanılacaksa kullanıldıktan sonra evaluate edilir ona göre karar verilir.


#%%
""" K=4 olarak karar verip görselleştirme yapalım: """

kmeans =  KMeans(n_clusters=4, init='k-means++')
y_pred = kmeans.fit_predict(X) #hem eğit hem de eğittikten sonra tahmin ettir. Tahminden kasıt hangi data hangi grupta.


plt.scatter(X[y_pred==0].Hacim, X[y_pred==0].Yas, s=100) 
plt.scatter(X[y_pred==1].Hacim, X[y_pred==1].Yas, s=100) 
plt.scatter(X[y_pred==2].Hacim, X[y_pred==2].Yas, s=100) 
plt.scatter(X[y_pred==3].Hacim, X[y_pred==3].Yas, s=100) 
