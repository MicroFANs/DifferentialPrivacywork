"""
@author:FZX
@file:OLH.py
@time:2020/1/13 3:25 下午
"""

import matplotlib
matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import random
import matplotlib.pyplot as plt

# 关闭科学计数法显示
np.set_printoptions(suppress=True)


# Locally Differentially Private Protocols for Frequency Estimation
# 论文中的OLH算法

epsilon=2
g=int(np.exp(epsilon)+1)
print(g)

path='../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data=bf.readcsv(path)
print(user_data[0][0])
n=len(user_data)


"""
encoding
"""
encode=np.zeros((n,g))
x=[] # 这里面存的是
print(encode)
for i in range(n):
    value=user_data[i][0]
    index=bf.hash(value,g,i)
    encode[i][index]=1
    x.append(index)

print(encode)
print(x)
#bf.savecsv(encode,'../LDPMiner/dataset/kosarak/encode.csv')



"""
perturbing
"""
p=rpb.gen_probability(epsilon,n=g)
q=p/np.e**epsilon
print(p,q)

y=[]
for i in range(n):
    y.append(lhb.grr(p,x[i],g))
print(y)

"""
Aggregation
"""



