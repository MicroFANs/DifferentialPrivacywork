# -*- coding:utf-8 -*-
"""
@author:FZX
@file:kmeans_numpy.py
@time:2019/3/7 14:16
"""
import numpy as np
import random
import pandas as pd
import sys

# 数据
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/s1_normal.csv',header=None)
dataset=[]
dataset=np.array(data)



# 欧氏距离
def distance(x1,x2):
    result=x1-x2
    return np.sqrt(np.sum(np.square(result)))

# 初始化质心
def center(data,k):
    # 随机初始化
    # center_point=random.sample(range(data.shape[0]),k)
    # center_array=data[center_point]

    # 分成k块初始化
    block=int(data.shape[0]/k)
    x = 0
    list=np.zeros((k,data.shape[1]))
    for i in range(k):
        list[i:]=data[x]
        x=x+block
    center_array=list
    return center_array

# kmeans iters为迭代次数，默认为20次迭代
def kmeans(data,k,iters=20):
    center_array=center(data,k)
    clusterchanged=True
    N=0 # 记录迭代次数
    temp = np.zeros(dataset.shape[0])
    # 收敛条件
    minsse= 10000
    while (clusterchanged and N<iters+1):
        clusterchanged=False
        print(N)
        for i in range(data.shape[0]):
            dis=[distance(data[i,:],center_array[j,:]) for j in range(k)]
            index=np.argmin(dis)
            #if temp[i]!=index:clusterchanged=True
            temp[i]=index

        for j in range(k):
            temp_res=data[temp==j]
            x1=np.mean(temp_res[:,0])
            x2=np.mean(temp_res[:,1])
            # x3=np.mean(temp_res[:,2])
            # x4=np.mean(temp_res[:,3])
            # x5 = np.mean(temp_res[:, 4])
            # x6 = np.mean(temp_res[:, 5])
            # x7 = np.mean(temp_res[:, 6])
            # x8 = np.mean(temp_res[:, 7])
            # x9 = np.mean(temp_res[:, 8])
            # x10 = np.mean(temp_res[:, 9])

            center_array[j,:]=[x1,x2]
            # center_array[j,:]=[x1,x2,x3,x4]
           # center_array[j, :] = [x1, x2,x3,x4,x5,x6,x7,x8,x9,x10]
            print('第'+str(N)+'次迭代的第'+str(j)+'簇质心:',center_array[j])

        # 计算簇内误差平方和 用来判断收敛
        # for j in range(k):
        #     temp_res = data[temp == j]
        #     cen=center_array[j,:]
        #     se=distance(temp_res,cen)
        # sse=np.sum(se)
        sse = 0
        for j in range(k):
            temp_res = data[temp == j]
            cen = center_array[j]
            se = distance(temp_res, cen)
            se2 = np.square(se)
            sse = sse + se2
        print('SSE:', sse)
        if (minsse-sse)>0.1:
            clusterchanged=True
            minsse=sse
        N=N+1

    km=np.c_[data,temp]
    return km  # temp是ndarray标签,km是原数据+标签


tp=kmeans(dataset,k=15,iters=40)
savefile = pd.DataFrame(tp)
print(tp)
print('y：保存，n：退出')
putin=input()
if putin=='y':
    savefile.to_csv('D:\Git\DifferentialPrivacywork\experiment2\output/S1/s1result_normal.csv',header=False,index=False)
elif putin=='n':
    sys.exit()




