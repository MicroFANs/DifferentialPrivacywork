# -*- coding:utf-8 -*-
"""
@author:FZX
@file:sklearnkmeans.py
@time:2019/3/16 16:06
"""
from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd

# 数据
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/s1_normal.csv',header=None)
dataset=[]
dataset=np.array(data)

kmeans=KMeans(n_clusters=15,init='random')
kmeans.fit(dataset)
sse=kmeans.inertia_
temp=kmeans.labels_
result=np.c_[dataset,temp]
print(result)
print('SSE:',sse)
savefile = pd.DataFrame(result)
#savefile.to_csv('D:\Git\DifferentialPrivacywork\experiment2\output\S1_sklkmeans1.csv',header=False,index=False)
