# -*- coding:utf-8 -*-
"""
@author:FZX
@file:myDPkmeans.py
@time:2019/3/15 20:30
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
dataset=[]
dataset=np.array(data)


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

# 计算每次迭代的最小预算
def mineps(k,N,d):
    minepslion=np.sqrt(((500*k**3)/N**2)*np.power((d+(4*d*0.45**2)**(1/3)),3))
    result=round(minepslion,3)
    return result

# 计算迭代次数
def rounds(mineps,epslion):
    # N=(epslion/mineps,10)
    # iters=np.int(N)
    # if iters<=2:
    #     iters=3
    # N=np.int(epslion/mineps)
    # if N>=7:
    #     iters=7
    # else:iters=N
    iters=10
    return iters

# 计算每一次迭代的隐私预算
def eacheps(iters,epslion,mineps):
    #d=(2/(iters-1))*((epslion/iters)-mineps)
    d=(2*epslion-2*mineps*iters)/(iters*(iters-1))
    eps=np.zeros(iters)
    for n in range(iters):
        e=mineps+n*d
        eps[n]=e
    each=eps[::-1]
    # if d<0:
    #     each=eps[::-1]
    # else:each=eps
    return each

# kmeans iters为迭代次数
def DPkmeans(data,k,epslion=6):
    sensitivity = dataset.shape[1] + 1  # 数据维数为d，敏感度为d+1
    minepslion=mineps(k,data.shape[0],data.shape[1]) # 求最小预算
    iters=rounds(minepslion,epslion) # 求迭代轮数
    print(iters)
    eachepslion=eacheps(iters,epslion,minepslion) # 分配每次迭代的隐私预算 等差数列法分配隐私预算
    print(eachepslion)
    clusterchanged=True # 收敛flag
    N=0 #记录迭代次数
    minsse=100000
    temp=np.zeros(data.shape[0])
    center_array=center(data,k)
    center_array_noise=center_array #　初始点不能加噪
    print('初始点:',center_array_noise,'\n')

    #for n in range(iters):
    while (clusterchanged and N<iters):
        clusterchanged=False
        print(N)
        epsofrounds=eachepslion[N]
        print('epslion:', epsofrounds)
        for i in range(data.shape[0]):
            dis=[distance(data[i,:],center_array_noise[j,:]) for j in range(k)]
            index=np.argmin(dis)  #  取使dis最小时的i
            temp[i]=index
        for j in range(k):
            temp_res=data[temp==j]
            num = temp_res.shape[0]
            noise0=laplacenoise(sensitivity,epsofrounds,1)
            num_noise=num+noise0[0]

            sum1=np.sum(temp_res[:,0]) # sum的格式为float64
            noise1=laplacenoise(sensitivity,epsofrounds,1)
            sum1_noise=sum1+noise1[0].astype('float64')
            x1=sum1_noise/num_noise
            # if x1 < 0:
            #     x1 = 0
            # elif x1 > 1:
            #     x1 = 1

            sum2=np.sum(temp_res[:,1])
            noise2 = laplacenoise(sensitivity,epsofrounds, 1)
            sum2_noise = sum2 + noise2[0].astype('float64')
            x2=sum2_noise/num_noise
            # if x2 < 0:
            #     x2 = 0
            # elif x2 > 1:
            #     x2 = 1

            sum3 = np.sum(temp_res[:, 2])
            noise3 = laplacenoise(sensitivity,epsofrounds, 1)
            sum3_noise = sum3 + noise3[0].astype('float64')
            x3 = sum3_noise / num_noise
            # # if x3 < 0:
            # #     x3 = 0
            # # elif x3 > 1:
            # #     x3 = 1
            #
            # sum4 = np.sum(temp_res[:, 3])
            # noise4 = laplacenoise(sensitivity,epsofrounds, 1)
            # sum4_noise = sum4 + noise4[0].astype('float64')
            # #print(sum4, '+', noise4, '=', sum4_noise)
            # #if sum4 == 0: print('sum4:warning\n')
            # x4 = sum4_noise / num_noise
            # # if x4 < 0:
            # #     x4 = 0
            # # elif x4 > 1:
            # #     x4 = 1

           # center_array_noise[j,:]=[x1,x2,x3,x4]
            center_array_noise[j,:]=[x1,x2,x3]
            print('第'+str(N)+'次迭代第'+str(j)+'簇的中心：',center_array_noise[j])


        # 计算簇内误差平方和 用来判断收敛
        sse=0
        for j in range(k):
            temp_res = data[temp == j]
            cen = center_array[j]
            se= distance(temp_res, cen)
            se2=np.square(se)
            sse=sse+se2
        print('SSE:',sse)
        print(minsse-sse)
        if (minsse - sse) > 0.1:
            clusterchanged = True
            minsse = sse
        N = N + 1
        print('============================================================================')
    km=np.c_[data,temp] # 将原数据和标签结合
    filename = 'D:\Git\DifferentialPrivacywork\experiment2/output/h8/h8myDP' + str(epslion) + '_' + str(N-1) + 'iters' + '.csv'
    return km ,filename,temp # temp是ndarray标签,km是原数据+标签

# # 获取当前日期作为文件名
# def name_time():
#     now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
#     filename='D:\Git\DifferentialPrivacywork\experiment2/output/'+now+'myDPBlood_out.csv'
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



tp,filename,label=DPkmeans(dataset,k=3,epslion=2)
# load对照数据
kmeansdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8result_normal.csv',header=None)
index=kmeansdata.shape[1]
kmeans=np.array(kmeansdata[index-1])
savefile = pd.DataFrame(tp)
print(tp)
print('====================指标======================')
measure(kmeans,label)
print('y：保存，n：退出')
putin=input()
if putin=='y':
    savefile.to_csv(filename,header=False,index=False)
elif putin=='n':
    sys.exit()







