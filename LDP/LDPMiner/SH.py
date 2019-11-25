"""
@author:FZX
@file:SH.py
@time:2019/11/25 20:25
"""

import numpy as np
import LDP.basicDP.basic_DP as bdp
import LDP.basicDP.mybasic as mdp
import LDP.basicDP.SHbasic as shb

'''
n：用户数量
d：项集中所有候选项数量
l：每个用户拥有的项集数量
j：项的编号，1<=j<=d
f_j：第j项的频率
k：top-k的heavy hitters
'''


input=[1,1,0,1,0,1,0,1,0]

# m-bit string x
x=shb.gen_m_bit_string(input)
print(x)

