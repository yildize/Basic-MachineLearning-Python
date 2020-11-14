# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:39:57 2019

@author: Erdo
"""

#%%
""" Önceki kodda Log. Reg. detaylıca yazılmıştı. Zaten bunu biliyordum ben. """
""" Şimdi burada Sci-kit Learn kütüphanesi kullanılarak, nasıl bir log. regressor oluştururuz onu göreceğiz:"""

# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% read csv
data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis
x_data = data.drop(["diagnosis"],axis=1)

# %% normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# (x - min(x))/(max(x)-min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

y_train = y_train.to_frame()
y_test = y_test.to_frame()

#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train,y_train)
print("Test accuracy {}", format(lr.score(x_test,y_test)))


