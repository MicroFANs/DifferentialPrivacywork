# -*- coding:utf-8 -*-
"""
@author:FZX
@file:svd.py
@time:2019/3/29 9:10
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def readpgm(name):    # 读取图片
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2'

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])   # 读取数据

    return (np.array(data[3:]),(data[1],data[0]),data[2])

data = readpgm('D:\Git\DifferentialPrivacywork\imagedata/faces/an2i/an2i_left_angry_open.pgm')   # 返回值data[0]为数据，data[1]为shape，data[2]为出现的最大数据。

print(data)
plt.imshow(np.reshape(data[0],data[1])) #
plt.show()

