"""
@author:FZX
@file:PrivKV.py
@time:2020/2/19 18:57
"""
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.PrivKVbasic as prkv

# windows路径
# path="/Workplace\pyworkplace\DifferentialPrivacywork\dataset\KV\privkvdata.txt"

# mac路径
# path="/Users/microfans/Documents/GitHub/DifferentialPrivacywork/dataset/KV/privkvdata.txt"


path = '../LDPdataset/KV/privkvdata.txt'

kvdata = bf.readtxt(path)

k = [d[0] for d in kvdata]  # list 装k值
v = [d[1] for d in kvdata]  # list 装v值

# # 用元组为元素构建kvlist，每个组是一个用户
# k_v=list(zip(k,v))
# print(k_v)

f, m = prkv.PrivKV(k, v, 10, 10)
print(f, m)

f_true, m_true = prkv.gettrue(k, v)
print(f_true, m_true)
