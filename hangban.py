# -*- coding:utf-8 -*-
"""
@author:FZX
@file:hangban.py
@time:2019/5/20 23:46
"""

from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd

# 数据
data=pd.read_excel('D:/hangban.xlsx',header=None)
dataset=[]
dataset=np.array(data)

kmeans=KMeans(n_clusters=10,init='random')
kmeans.fit(dataset)
sse=kmeans.inertia_
temp=kmeans.labels_
result=np.c_[dataset,temp]
print(result)

savefile = pd.DataFrame(result)
savefile.to_csv('D:\hangban.csv',header=False,index=False)