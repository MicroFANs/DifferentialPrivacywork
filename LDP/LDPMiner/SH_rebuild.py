"""
@author:FZX
@file:SH_rebuild.py
@time:2020/1/16 9:58 下午
"""


import matplotlib
matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# 关闭科学计数法显示
np.set_printoptions(suppress=True)






# 读取数据
path='../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data=bf.readcsv(path)
print('data:\n',user_data)
max = np.max(user_data) # 一维数据中的最大值
min = np.min(user_data) # 一维数据中的最小值
n=len(user_data) # 用户i的数量n
d=max-min+1 # 用户数据域的维度d
print('n=', n, '\nd=', d)
label = np.unique(user_data)
x_list = label.tolist()




"""
原始数据，用来计算误差
"""
list_true=bf.csvtolist(path) # list_true是真实的数据list
array_true=np.array(list_true)
print(array_true)

label = np.unique(user_data)
x_list = label.tolist()



epsilon=2

# 计算随机投影矩阵参数
r,m=shb.comput_parameter(d=d,n=8,epsilon=epsilon,beta=0.01)
print('m=',m)
rnd_proj=shb.genProj(m,d)


#server
z_total=np.zeros(m)
for i in range(n):
    x=int(array_true[i])
    #print(x)
    z=shb.Basic_Randomizer_1(m=m,x=x,epsilon=epsilon,rand_proj=rnd_proj)
    #print(z)
    z_total=z_total+z

print(z_total)


"""
计算估计值
"""
count_true=[] #真实计数结果
count_estimate=[] # 估计计数结果
for i in range(d):
    count_true.append(list_true.count(x_list[i]))

    estimate=shb.FO(rnd_proj,z_total,x_list[i])
    count_estimate.append(estimate)


print(count_true) #真实的计数结果
print(count_estimate) # 估计的计数结果


"""
保存结果
"""
result={"SH":count_estimate,"true":count_true,"value":x_list}
df_res=pd.DataFrame(result)
df_res.to_csv("../LDPMiner/dataset/kosarak/result/10k_sv_SH_rb.csv",index=False)
