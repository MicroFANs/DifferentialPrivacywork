"""
@author:FZX
@file:sampling_RP.py
@time:2019/12/11 15:40
"""
import numpy as np
import pandas as pd
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.RPbasic as rpb

'''
复现LDPMiner论文中的sampling RAPPOR方法

n：用户数量
d：项集中所有候选项数量
j：项的编号，1<=j<=d
f_j：第j项的频率
'''

# 读取数据
path='../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data=bf.readcsv(path)
print('data:\n',user_data)
max = np.max(user_data) # 一维数据中的最大值
min = np.min(user_data) # 一维数据中的最小值
n=len(user_data) # 用户i的数量n
d=max-min+1 # 用户数据域的维度d