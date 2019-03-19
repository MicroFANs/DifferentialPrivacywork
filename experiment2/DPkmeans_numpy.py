# -*- coding:utf-8 -*-
"""
@author:FZX
@file:DPkmeans_numpy.py
@time:2019/3/7 16:30
"""
import numpy as np
import random
import pandas as pd
import time
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# 数据
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset\h8_normal.csv',header=None)
#data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/testSet.csv',header=None)
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
# def center(data,k):
#     center_point=random.sample(range(data.shape[0]),k)
#     center_array=data[center_point]
#     return center_array
def center(data,k):
    # 分成k块初始化
    block=int(data.shape[0]/k)
    x = 0
    list=np.zeros((k,data.shape[1]))
    for i in range(k):
        list[i:]=data[x]
        x=x+block
    center_array=list
    return center_array

# laplace噪声
def laplacenoise(sensitivity,epslion,len):  #  产生单个laplace噪声
    location=0
    scale=sensitivity/epslion
    Laplacian_noise =np.random.laplace(location, scale, len)
    return Laplacian_noise # 格式为ndarray
# laplace_array
def laplacenoise_array(sensitivity,epslion,len,num):  #  产生laplace噪声数组,len是维数，num是生成的个数
    location=0
    scale=sensitivity/epslion
    list=[]
    for i in range(num):
        list .append( np.random.laplace(location, scale, len))
        Laplacian_noise=np.array(list)
    return Laplacian_noise

# kmeans iters为最大迭代次数，默认为10次迭代
def DPkmeans(data,k,iters=10,epslion=6,allocation=0 or 1):
    sensitivity=dataset.shape[1]+1 # 数据维数为d，敏感度为d+1
    eps=epslion
    if allocation==0:
        allocat='avg' # 用于文件名
        epslion=epslion/iters # 平均分隐私预算
        print('avgepslion:',epslion)
    center_array=center(data,k)
    center_array_noise=center_array #　初始点不能加噪
    print('初始点:',center_array_noise,'\n')

    #for n in range(iters):
    clusterchanged=True
    N=0 # 记录迭代次数
    temp = np.zeros(dataset.shape[0])
    # 收敛条件
    minsse = 10000
    while (clusterchanged and N<iters+1):
        clusterchanged=False
        print(N)
        for i in range(data.shape[0]):
            dis=[distance(data[i,:],center_array_noise[j,:]) for j in range(k)]
            index=np.argmin(dis)  #  取使dis最小时的i
            temp[i]=index
        for j in range(k):
            temp_res=data[temp==j]
            num = temp_res.shape[0]
            noise0=laplacenoise(sensitivity,epslion,1)
            num_noise=num+noise0[0]

            sum1=np.sum(temp_res[:,0]) # sum的格式为float64
            noise1=laplacenoise(sensitivity,epslion,1)
            sum1_noise=sum1+noise1[0].astype('float64')
            x1=sum1_noise/num_noise

            sum2=np.sum(temp_res[:,1])
            noise2 = laplacenoise(sensitivity, epslion, 1)
            sum2_noise = sum2 + noise2[0].astype('float64')
            x2=sum2_noise/num_noise
            # if x2 < 0:
            #     x2 = 0
            # elif x2 > 1:
            #     x2 = 1

            sum3 = np.sum(temp_res[:, 2])
            noise3 = laplacenoise(sensitivity, epslion, 1)
            sum3_noise = sum3 + noise3[0].astype('float64')
            x3 = sum3_noise / num_noise
            # # if x3 < 0:
            # #     x3 = 0
            # # elif x3 > 1:
            # #     x3 = 1
            #
            # sum4 = np.sum(temp_res[:, 3])
            # noise4 = laplacenoise(sensitivity, epslion, 1)
            # sum4_noise = sum4 + noise4[0].astype('float64')
            # #print(sum4, '+', noise4, '=', sum4_noise)
            # #if sum4 == 0: print('sum4:warning\n')
            # x4 = sum4_noise / num_noise
            # # if x4 < 0:
            # #     x4 = 0
            # # elif x4 > 1:
            # #     x4 = 1

            #center_array_noise[j,:]=[x1,x2,x3,x4]
            center_array_noise[j,:]=[x1,x2,x3]
            print('第'+str(N)+'次迭代第'+str(j)+'簇的中心：',center_array_noise[j])
        # 收敛条件 SSE<0.1
        sse=0
        for j in range(k):
            temp_res = data[temp == j]
            cen=center_array_noise[j]
            se=distance(temp_res,cen)
            se2=np.square(se)
            sse=sse+se2
        print('SSE:',sse)
        #if np.abs(minsse - sse) > 0.1:
        print(minsse-sse)
        if abs(minsse - sse) > 0.8:
            clusterchanged = True
            minsse = sse
        N=N+1
        if allocation==1:
            allocat = 'div2'  # 用于文件名
            epslion=epslion/2 # 二分法分配隐私预算
            print('1/2epslion:',epslion)
        print('============================================================================')
    km=np.c_[data,temp] # 将原数据和标签结合

    filename='D:\Git\DifferentialPrivacywork\experiment2/output/h8/h8'+allocat+'DP'+str(eps)+'_'+str(N-1)+'iters'+'.csv'
    return km,filename,temp # temp是ndarray标签,km是原数据+标签

# # 获取当前日期作为文件名
# def name_time():
#     now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
#     filename='D:\Git\DifferentialPrivacywork\experiment2/output/S1/'+now+'DP_out.csv'
#     return filename

# 指标
def measure(y_true,y_pred):
    # 混淆矩阵
    confmat=confusion_matrix(y_true,y_pred)
    print(confmat)
    # #plt.matshow(confmat, cmap=plt.cm.gray)
    # #plt.show()
    #
    # fig,ax = plt.subplots(figsize=(7,7))
    # cax=ax.matshow(confmat,cmap=plt.cm.Blues,alpha=1)
    # #fig.colorbar(cax)
    # for i in range(confmat.shape[0]):
    #     for j in range(confmat.shape[1]):
    #         ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
    # plt.xlabel('predicted label')
    # plt.ylabel('true label')
    # plt.title('confusion matrix')
    # plt.show()
    measurelist=classification_report(y_true=y_true, y_pred=y_pred)
    print(measurelist)

'''
=======================================================================================
'''


# allocation :0 是均匀分配；1是二分法
tp,filename,label=DPkmeans(dataset,k=3,iters=10,epslion=2,allocation=1)

# load对照数据
kmeansdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8result_normal.csv',header=None)
index=kmeansdata.shape[1]
kmeans=np.array(kmeansdata[index-1])
savefile = pd.DataFrame(tp)
print(tp)
print('=================指标===================')
measure(kmeans,label)
print('y：保存，n：退出')
putin=input()
if putin=='y':
    savefile.to_csv(filename,header=False,index=False)
elif putin=='n':
    sys.exit()