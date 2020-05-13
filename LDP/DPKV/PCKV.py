"""
@author:FZX
@file:PCKV.py
@time:2020/2/19 18:57
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

# 关闭科学计数法显示
np.set_printoptions(suppress=True)

data_k = bf.readtxt('../LDPdataset/KV/KV_k.txt')
data_v = bf.readtxt('../LDPdataset/KV/KV_v.txt')


#print(data_k[0],'\n',data_v[0])


def example(k, v):
    return k + 1, v + 1


# 这样写生成的是元素为元组的list
kvp = [example(data_k[i][0], data_v[i][0]) for i in range(len(data_k))]
print(kvp)

# 构建以元组元素的list
kv = zip(data_k[0], data_v[0])
#print(list(kv))
