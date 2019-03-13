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
# 数据
data=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset/Iris_normal.csv',header=None)
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
def center(data,k):
    center_point=random.sample(range(data.shape[0]),k)
    center_array=data[center_point]
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

# kmeans iters为迭代次数，默认为20次迭代
def DPkmeans(data,k,iters=4,epslion=6):
    sensitivity=dataset.shape[1]+1 # 数据维数为d，敏感度为d+1
    #sensitivity=1

    epslion=epslion/iters # 平均分隐私预算

    temp=np.zeros(data.shape[0])
    center_array=center(data,k)
    # center_noise=laplacenoise_array(1,0.5,2,k)
    # center_array_noise=center_array+center_noise  # 终于找到问题在哪了，如果把数据0-1归一化，一旦添加噪声后的初始点出了这个0-1的范围，你们它将永远不会有点与它最近，即没有点会分到它的簇中去
    center_array_noise=center_array #　初始点不能加噪

    print('初始点',center_array_noise)
    for n in range(iters):
        for i in range(data.shape[0]):
            dis=[distance(data[i,:],center_array_noise[j,:]) for j in range(k)]
            index=np.argmin(dis)  #  取使dis最小时的i
            temp[i]=index
        for j in range(k):
            temp_res=data[temp==j]
            num = temp_res.shape[0]
            # print('temp',temp_res)
            # print(temp_res.shape)
            # x1=np.mean(temp_res[:,0])
            # x2=np.mean(temp_res[:,1])

            sum1=np.sum(temp_res[:,0]) # sum的格式为float64
            noise1=laplacenoise(sensitivity,epslion,1)
            sum1_noise=sum1+noise1[0].astype('float64')
            print(sum1,'+',noise1,'=',sum1_noise)
            #if sum1==0:print('sum1:warning\n')
            if sum1_noise<0:
                sum1_noise==0
            elif sum1_noise>1:
                sum1_noise==1
            x1=sum1_noise/num

            sum2=np.sum(temp_res[:,1])
            noise2 = laplacenoise(sensitivity, epslion, 1)
            sum2_noise = sum2 + noise2[0].astype('float64')
            print(sum2, '+', noise2, '=', sum2_noise)
            #if sum2==0:print('sum2:warning\n')
            if sum2_noise < 0:
                sum2_noise == 0
            elif sum2_noise > 1:
                sum2_noise == 1
            x2=sum2_noise/num

            sum3 = np.sum(temp_res[:, 2])
            noise3 = laplacenoise(sensitivity, epslion, 1)
            sum3_noise = sum3 + noise3[0].astype('float64')
            print(sum3, '+', noise3, '=', sum3_noise)
            #if sum3 == 0: print('sum3:warning\n')
            if sum3_noise < 0:
                sum3_noise == 0
            elif sum3_noise > 1:
                sum3_noise == 1
            x3 = sum3_noise / num

            sum4 = np.sum(temp_res[:, 3])
            noise4 = laplacenoise(sensitivity, epslion, 1)
            sum4_noise = sum4 + noise4[0].astype('float64')
            print(sum4, '+', noise4, '=', sum4_noise,'\n')
            #if sum4 == 0: print('sum4:warning\n')
            if sum4_noise < 0:
                sum4_noise == 0
            elif sum4_noise > 1:
                sum4_noise == 1
            x4 = sum4_noise / num

            center_array_noise[j,:]=[x1,x2,x3,x4]
            print('第'+str(n)+'次迭代第'+str(j)+'簇的中心：',center_array_noise[j],'\n')
        print('epslion=',epslion)

        #epslion=epslion/2 # 二分法分配隐私预算

        print('----------------------------')
    km=np.c_[data,temp] # 将原数据和标签结合
    return km  # temp是ndarray标签,km是原数据+标签

# 获取当前日期作为文件名
def name_time():
    now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    filename='D:\Git\DifferentialPrivacywork\experiment2/output/'+now+'DPIirs_out.csv'
    return filename












# cent=center(dataset,4)
# print(cent)
# x=laplacenoise_array(1,0.5,2,4)
# print(cent+x)

tp=DPkmeans(dataset,3,iters=20,epslion=100)
filename=name_time()
savefile = pd.DataFrame(tp)
print(tp)
savefile.to_csv(filename,header=False,index=False)


# # 测试噪声函数
# # x=laplacenoise_array(1,0.5,2,80)
# # print(x)
# # print(type(x))
# # z=dataset+x
# # print(z)
# # print(kmeans(z,4))
