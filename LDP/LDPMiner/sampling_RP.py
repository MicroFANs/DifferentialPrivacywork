"""
@author:FZX
@file:sampling_RP.py
@time:2019/12/11 15:40
"""
# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.RPbasic as rpb
import matplotlib.pyplot as plt

# 关闭科学计数法显示
np.set_printoptions(suppress=True)

'''
复现LDPMiner论文中的sampling RAPPOR方法

n：用户数量
d：项集中所有候选项数量
l：用户项集的长度
j：项的编号，1 in [1,l]
f_j：第j项的频率
'''

# 读取数据
path='../LDPMiner/dataset/kosarak/kosarak_10k.txt'
user_data=bf.readtxt(path)


# all_iterm是存放所有出现过的项的数组，用来查找项对于的index，以便进行编码
iterm_list=bf.combine_lists(user_data)
label=np.unique(np.array(iterm_list))
x_list=label.tolist()
print(label)

n=len(user_data)
d=len(label)
l=10


"""
user
"""

v_list=bf.gen_iterm_set(user_data,l)
print(v_list)

sampling_data=np.zeros((n,1))
for i in range(n):
    one_hot=np.zeros(d+1)# 还要算上0，0是虚拟项
    sampling_data[i]=random.choice(v_list[i])

print(sampling_data)


onehotcode=bf.one_hot(sampling_data,label)
print(onehotcode)

epsilon=10# eps越大，数据可用性越好，隐私保护水平越低
f=rpb.gen_f(epsilon)

z_totle=np.zeros(d)
for i in range(n):
    z=rpb.PRR_bits(onehotcode[i],f)
    z_totle=z_totle+z
print(z_totle)

est=np.zeros(d)
for j in range(d):
    est[j]=rpb.decoder_PRR(z_totle[j],n,f)*l
print('estimate:\n',est)


# 画图
plt.bar(x_list,est)
plt.show()