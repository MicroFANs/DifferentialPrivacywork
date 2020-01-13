"""
@author:FZX
@file:OLH.py
@time:2020/1/13 3:25 下午
"""

import matplotlib
matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import random
import matplotlib.pyplot as plt

# 关闭科学计数法显示
np.set_printoptions(suppress=True)


# Locally Differentially Private Protocols for Frequency Estimation
# 论文中的OLH算法

epsilon=1
g=int(np.exp(epsilon)+1)
print(g)