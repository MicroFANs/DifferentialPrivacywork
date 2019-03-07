# -*- coding:utf-8 -*-
"""
@author:FZX
@file:DP-k-means.py
@time:2019/3/5 15:37
"""
from numpy import *
from sklearn import preprocessing
import numpy as np


def loadDataSettxt(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def loadcsv(filename):
    return 0

def normalization(list): #  0-1归一化
    dataset=np.array(list)
    min_max_scaler=preprocessing.MinMaxScaler()
    normaldataset=min_max_scaler.fit_transform(dataset)
    return normaldataset

def distEclud(vecA, vecB):# 计算欧氏距离
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k): # 获取随机初始点
    n = shape(dataSet)[1] # n是有多少列
    centroids = mat(zeros((k,n)))#create centroid mat k行n列
    for j in range(n):#create random cluster centers, within bounds of each dimension 有n个纬度，即n个列
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
def myrandCent(dataSet,k):
    m=shape(dataSet)[0]#行数
    n=shape(dataSet)[1]#维度、列数
    centroids=dataSet.take(np.random.choice(m,k),axis=0)
    lap=laplacenoise_array(1,0.5,n,k)
    centroids_noise=centroids+lap
    #return centroids,lap,centroids_noise
    return centroids_noise

def laplacenoise(sensitivity,epslion,len):  #  产生单个laplace噪声
    location=0
    scale=sensitivity/epslion
    Laplacian_noise =np.random.laplace(location, scale, len)
    return Laplacian_noise

def laplacenoise_array(sensitivity,epslion,len,num):  #  产生laplace噪声数组,len是维数，num是生成的个数
    location=0
    scale=sensitivity/epslion
    list=[]
    for i in range(num):
        list .append( np.random.laplace(location, scale, len))
        Laplacian_noise=np.array(list)
    return Laplacian_noise

def kMeans(dataSet, k, distMeas=distEclud, createCent=myrandCent,maxiter=10):
    m = shape(dataSet)[0] #行数
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points存放样本属于的类和质心距离SE
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    iterCount=0
    clusterChanged = True
    while clusterChanged and iterCount<maxiter:# 设置最大轮数，默认为10
        iterCount+=1
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True#如果分配发生变化则要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2
        #print (centroids)# 一次迭代后的中心 <class 'numpy.matrixlib.defmatrix.matrix'>格式
        for cent in range(k):#recalculate centroids
            pointsInCluster= dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster取clusterAssment中第一列为cent的点，即取当前簇为cent的点
            print('cent',cent,pointsInCluster)
            if len(pointsInCluster) != 0:
                centroids[cent,:] = mean(pointsInCluster, axis=0) #assign centroid to mean
    return centroids, clusterAssment

# 测试主程序
dataMat=loadDataSettxt('D:\Git\DifferentialPrivacywork\dataset/testSet.txt')
print(type(dataMat))
print(dataMat)
# normaldataset=normalization(dataMat)
# # print(normaldataset)
# x=normaldataset.tolist()
# print(type(normaldataset))
# print(type(x))
# print(len(x))
# print(x)
myCentroids,clustAssing = kMeans(mat(dataMat),4)
print(clustAssing)

# 为什么归一化之后就没用了？缺值，不归一化就没事。

# #测试laplace
# x=[]
# for i in range(4):
#     x.append(laplacenoise(1,0.5,2))
# y=np.array(x)
# print(y)
# # 测试laplacenoise_array()
# y=laplacenoise_array(1,0.5,2,4)
# print(y)

# 测试myrandcent
# dataset=loadDataSettxt('D:\Git\DifferentialPrivacywork\dataset/testSet.txt')
# x,y,z=myrandCent(mat(dataset),4)
#
# # print(x)
# # # print(y)
# print(z)
# print(type(z))
# y=randCent(mat(dataset),3)
# print(type(y))

# # 测试加噪
# dataset=loadDataSettxt('D:\Git\DifferentialPrivacywork\dataset/testSet.txt')
# x1=myrandCent(mat(dataset),4)
# x1=np.array(x1)
# print(x1)
# n=y+x1
# print(n)