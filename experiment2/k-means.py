# -*- coding:utf-8 -*-
"""
@author:FZX
@file:k-means.py
@time:2019/2/28 16:37
"""
from numpy import *
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
    m=shape(dataMat)[0]#行数
    n=shape(dataMat)[1]#维度、列数
    centriods=[] # 创建list用来装中心点
    first=[] #创建list用来存k个随机数，用来找随机的k个初试点
    for i in range(k):
        first=random.randint(0,m-1)# 随机数在0-m-1之间
        centriods.append(dataMat[first])
    return mat(centriods)



def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #行数
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points存放样本属于的类和质心距离SE
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
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
            centroids[cent,:] = mean(pointsInCluster, axis=0) #assign centroid to mean
    return centroids, clusterAssment

dataMat=loadDataSettxt('D:\Git\DifferentialPrivacywork\dataset/testSet.txt')
# print(dataMat)
# print('随机点：')
#
# print(l)
# x=randCent(mat(dataMat),5)
# print(type(x))
# print(x)
#
# y=myrandCent(dataMat,10)
# print(type(y))
# print(distEclud(y[0],y[1]))
#
# z=kMeans(mat(dataMat),4)
myCentroids,clustAssing = kMeans(mat(dataMat),4)
# print (myCentroids)
# print (clustAssing)
