"""
@author:FZX
@file:hitrate.py
@time:2020/9/8 7:43 下午
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import random
import time

"""数据处理"""
# 关闭科学计数法显示
np.set_printoptions(suppress=True)


n=40

Clothing='../LDPdataset/Clothing/data/Clothing_id_count.txt'
Ecomm='../LDPdataset/E_Commerce/data/Ecomm_id_count_avg.txt'
Movie='../LDPdataset/Movie/data/Movie_id_count_avg.txt'

Clothingtopk='../LDPdataset/Clothing/result/Clothing_realtopk.txt'
Ecommtopk='../LDPdataset/E_Commerce/result/Ecomm_realtopk.txt'
Movietopk='../LDPdataset/Movie/result/Movie_realtopk.txt'


data=bf.readtxt(Clothing)
data=sorted(data,key=lambda x:x[1],reverse=True)
realtopk=[]
for i in range(n):
    realtopk.append(int(data[i][0]))
bf.savetxt(realtopk,Clothingtopk)






