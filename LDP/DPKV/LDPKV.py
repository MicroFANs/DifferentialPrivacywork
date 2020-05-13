"""
@author:FZX
@file:LDPKV.py
@time:2020/5/12 8:41 下午
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import random


# 关闭科学计数法显示
np.set_printoptions(suppress=True)

data_k = bf.readtxt('../LDPdataset/KV/KV_k.txt')
data_v = bf.readtxt('../LDPdataset/KV/KV_v.txt')

#用户数量
n=len(data_k)

#构建kv对元组列表，每一行代表一个用户的kv对list
kv=[]
for i in range(n):
    tmp=zip(data_k[i],data_v[i])
    kv.append(list(tmp))


#通过采样（每个用户随机抽取一个kv对），构建上传upkvlist，长度为用户数目n，表示每个用户上传对kv对
upkv=[]
for i in range(n):
    tmp=random.sample(kv[i],1)
    upkv.append(tmp[0])
print(upkv)


#调用OLH






