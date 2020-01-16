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
import pandas as pd
import random
import matplotlib.pyplot as plt

# 关闭科学计数法显示
np.set_printoptions(suppress=True)


# Locally Differentially Private Protocols for Frequency Estimation
# 论文中的OLH算法

epsilon=2
g=int(np.exp(epsilon)+1)
#print(g)

path='../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data=bf.readcsv(path)
#print(user_data)
n=len(user_data)
#print(n)




"""
原始数据，用来计算误差
"""
list_true=bf.csvtolist(path) # list_true是真实的数据list
#print(list_true)

label = np.unique(user_data)
x_list = label.tolist()






"""
encoding
"""
encode=np.zeros((n,g))
x=[] # 这里面存的是
#print(encode)
for i in range(n):
    value=user_data[i][0]
    index=bf.hash(value,g,i)
    #encode[i][index]=1
    x.append(index)

# print(encode)
# print(x)
#bf.savecsv(encode,'../LDPMiner/dataset/kosarak/encode.csv')



"""
perturbing
"""
p=rpb.gen_probability(epsilon,n=g)
q=p/np.e**epsilon
#print(p,q)

y=[]
for i in range(n):
    y.append(lhb.grr(p,x[i],g))
#print(y)

"""
Aggregation
"""
# 单次查询结果
# v=11
# c=lhb.support(v,y,g,n)
# print(c)
#
# estimate=lhb.aggregation(c,n,g,p)
# print(estimate/n)


count_true=[] #真实计数结果
count_estimate=[] # 估计计数结果
for i in range(len(x_list)):
    count_true.append(list_true.count(x_list[i]))

    c=lhb.support(x_list[i],y,g,n)
    estimate=lhb.aggregation(c,n,g,p)
    count_estimate.append(estimate)

print(count_true) #真实的计数结果
print(count_estimate) # 估计的计数结果

"""
保存结果
"""
result={"OLH":count_estimate,"true":count_true,"value":x_list}
df_res=pd.DataFrame(result)
df_res.to_csv("../LDPMiner/dataset/kosarak/result/10k_sv_OLH.csv",index=False)

