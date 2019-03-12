# -*- coding:utf-8 -*-
"""
@author:FZX
@file:kmeans_numpy.py
@time:2019/3/7 14:16
"""
import numpy as np
import random
import pandas as pd

# 数据
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/Iris_normal.csv',header=None)
# print(data.shape)
# print(data[1])
dataset=[]
dataset=np.array(data)
#print(dataset)


# 欧氏距离
def distance(x1,x2):
    result=x1-x2
    return np.sqrt(np.sum(np.square(result)))

# 初始化质心
def center(data,k):
    center_point=random.sample(range(data.shape[0]),k)
    center_array=data[center_point]
    return center_array

# kmeans iters为迭代次数，默认为20次迭代
def kmeans(data,k,iters=20):
    temp=np.zeros(data.shape[0])
    center_array=center(data,k)
    for n in range(iters):
        for i in range(data.shape[0]):
            dis=[distance(data[i,:],center_array[j,:]) for j in range(k)]
            index=np.argmin(dis)
            temp[i]=index
        for j in range(k):
            temp_res=data[temp==j]
            x1=np.mean(temp_res[:,0])
            x2=np.mean(temp_res[:,1])
            x3=np.mean(temp_res[:,2])
            x4=np.mean(temp_res[:,3])
            center_array[j,:]=[x1,x2,x3,x4]
    km=np.c_[data,temp]
    return km  # temp是ndarray标签,km是原数据+标签


# cent=center(dataset,3)
# print(cent)
tp=kmeans(dataset,3)
savefile = pd.DataFrame(tp)
savefile.to_csv('D:\Git\DifferentialPrivacywork\experiment2/output/Irisresult_normal.csv',header=False,index=False)
print(tp)
