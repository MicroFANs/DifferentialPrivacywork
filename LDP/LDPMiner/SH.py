"""
@author:FZX
@file:SH.py
@time:2019/11/25 20:25
"""

import numpy as np
import pandas as pd
import LDP.basicDP.SHbasic as shb
import LDP.basicFunction.basicfunc as bf


'''
复现Local, Private, Efficient Protocols for Succinct Histograms论文中的基础方法，必须d>n

n：用户数量
d：项集中所有候选项数量
j：项的编号，1<=j<=d
f_j：第j项的频率
'''



# 读取数据
path='../LDPMiner/dataset/SH/kosarak_10k_singlevalue.csv'
user_data=bf.readcsv(path)
print('data:\n',user_data)
max = np.max(user_data) # 一维数据中的最大值
min = np.min(user_data) # 一维数据中的最小值
n=len(user_data) # 用户i的数量n
d=max-min+1 # 用户数据域的维度d


# onehot编码
onehot_data=bf.one_hot_1D(user_data)
#np.savetxt('../LDPMiner/dataset/SH/kosarak_10k_singlevalue_onehot.txt',onehot_data,fmt='%d')
print('数据obehot编码:\n',onehot_data)



# 下面用到投影矩阵的只适合于d>n的情况，这样子算出的m才小于d，才是降维


# 参数设置
#epsilon=np.log(3) # eps=ln(3)
epsilon=1
r,m=shb.comput_parameter(d=d,n=n,epsilon=epsilon,beta=0.05)
print('n=',n,'\nd=',d,'\nr=',r,'\nm=',m)



# 生成随机投影矩阵
rnd_proj=shb.genProj(m,d)
print('\nrnd_proj:\n',rnd_proj)

# sever端计算估计值
z_total=np.zeros(m)
for i in range(n):
    e=onehot_data[i]
    x=np.matmul(rnd_proj,e)
    z=shb.Basic_Randomizer(x,epsilon)
    z_total=z_total+z
z_mean=z_total/n

# 选择计算某个v的频率估计
v=8
e_v=onehot_data[v]
f_est=shb.Frequency_Estimator_Based_on_FO(rnd_proj,z_mean,e_v)
print('v=',v,'\nf_estmiate=',f_est)

count=sum(user_data==v)
print('f_true=',count[0]/n)



