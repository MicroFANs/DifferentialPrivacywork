# -*- coding:utf-8 -*-
"""
@author:FZX
@file:Data preprocessing.py
@time:2019/2/27 15:59
"""
from numpy import *
from sklearn import preprocessing
import numpy as np
import pandas as pd


def loadDataSettxt(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def normalization(list): #  0-1归一化
    dataset=np.array(list)
    min_max_scaler=preprocessing.MinMaxScaler()
    normaldataset=min_max_scaler.fit_transform(dataset)
    return normaldataset


# txt 文件
# dataMat=loadDataSettxt('D:\Git\DifferentialPrivacywork\dataset/testSet.txt')
# print(type(dataMat))
# print(dataMat)
# normaldataset=normalization(dataMat)
# print(normaldataset)
# print(type(normaldataset))
# np.savetxt('D:\Git\DifferentialPrivacywork\dataset/set.txt',normaldataset,fmt="%.8f",delimiter="\t") # 保留8位小数



# csv文件
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/Iris.csv',header=None)
dataset=[]
dataset=np.array(data)
normal=normalization(dataset)
savefile = pd.DataFrame(normal)
savefile.to_csv('D:\Git\DifferentialPrivacywork\dataset\Iris_normal.csv',header=False,index=False,float_format = '%.8f')
print(savefile)

