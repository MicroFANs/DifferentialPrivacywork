"""
@author:FZX
@file:myLDPAlgorithm.py
@time:2019/12/30 4:12 下午
"""

import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 读取数据
path='../LDPMiner/dataset/kosarak/kosarak.txt'
user_data=bf.readtxt(path)

# all_iterm是存放所有出现过的项的数组，用来查找项对于的index，以便进行编码
iterm_list=bf.combine_lists(user_data)
label=np.unique(np.array(iterm_list))
x_list=label.tolist() # 数据标签
print(x_list)



# d=len(label) # 获取数据纬度
# n=len(user_data) # 获取用户数

#kosarak.txt数据集的参数
d=41270
n=990002



# user




# server



# 评估