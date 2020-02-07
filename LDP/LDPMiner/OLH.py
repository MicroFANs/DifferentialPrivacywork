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
import time

# 关闭科学计数法显示
np.set_printoptions(suppress=True)

# Locally Differentially Private Protocols for Frequency Estimation
# 论文中的OLH算法



path = '../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data = bf.readcsv(path)
# print(user_data)
n = len(user_data)
# print(n)


epsilon = 2
g = int(np.exp(epsilon) + 1)
# print(g)

# 程序运行起始时间
starttime=time.clock()


"""
原始数据，用来计算误差
"""
list_true = bf.csvtolist(path)  # list_true是真实的数据list
# print(list_true)

label = np.unique(user_data)
x_list = label.tolist()

"""
encoding
"""
encode = np.zeros((n, g))
x = []  # 这里面存的是
# print(encode)
for i in range(n):
    value = user_data[i][0]

    # 自己写的hash函数
    #index = bf.hash(value, g, i)

    # 作者源码实现方式olh_hash,效果很差，不知道为什么。原因找到了，没有修改与之匹配的support函数
    """替换hash函数时一定别忘记更换下面匹配的olh_support函数，一共要改3处，hash，support，grr"""
    index = lhb.olh_hash(value, g, i)

    # encode[i][index]=1
    x.append(index)

# print(encode)
# print(x)
# bf.savecsv(encode,'../LDPMiner/dataset/kosarak/encode.csv')


"""
perturbing
"""
p = rpb.gen_probability(epsilon, n=g)
q = p / np.e ** epsilon
# print(p,q)

y = []
for i in range(n):

    # 自己实现的grr
    #y.append(lhb.grr(p, x[i], g))
    # 成作者源码实现方式olh_grr，但是作者实现的grr结果都一样，不会变，不知道为什么。答案是我自己的hash函数结果不会变，是因为用了系统时间种子
     y.append(lhb.olh_grr(p, x[i], g))

# print(y)

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


count_true = []  # 真实计数结果
count_estimate = []  # 估计计数结果
for i in range(len(x_list)):
    count_true.append(list_true.count(x_list[i]))

    # 自己写的support函数匹配自己的hash函数
    #c = lhb.support(x_list[i], y, g, n)
    # 与作者源码的olh_hash匹配的support函数
    """替换hash函数时一定别忘记更换下面匹配的olh_support函数"""
    c = lhb.olh_support(x_list[i], y, g, n)


    estimate = lhb.aggregation(c, n, g, p)
    count_estimate.append(estimate)

# 程序运行结束时间
endtime=time.clock()


print(count_true)  # 真实的计数结果
print(count_estimate)  # 估计的计数结果
print("运行时间：%s s"%(endtime-starttime))

"""
保存结果
"""
result = {"OLH": count_estimate, "true": count_true, "value": x_list}
df_res = pd.DataFrame(result)
df_res.to_csv("../LDPMiner/dataset/kosarak/result/10k_sv_OLH.csv", index=False)
