"""
@author:FZX
@file:RP.py
@time:2019/12/11 15:45
"""
import numpy as np
import LDP.basicDP.RPbasic as rpb
import LDP.basicFunction.basicfunc as bf
import matplotlib.pyplot as plt

# 关闭科学计数法显示
np.set_printoptions(suppress=True)

"""
复现RAPPOR中的方法，针对单值问题，即每个用户拥有一个值
"""

path = '../LDPMiner/dataset/kosarak/kosarak_10k_singlevalue.csv'
user_data = bf.readcsv(path)
print('data:\n', user_data)

max = np.max(user_data)  # 一维数据中的最大值
min = np.min(user_data)  # 一维数据中的最小值
n = len(user_data)  # 用户i的数量n
d = max - min + 1  # 用户数据域的维度d
print('n=',n,'\nd=',d)
label=np.unique(user_data)



# onehot编码
onehot_data = bf.one_hot_1D(user_data)
# np.savetxt('../LDPMiner/dataset/SH/kosarak_10k_singlevalue_onehot.txt',onehot_data,fmt='%d')
#print('数据obehot编码:\n', onehot_data)


# 计算每个项的真实频数用于画图
origin=np.zeros(d)
for i in range(n):
    origin=origin+onehot_data[i]
print('origin:\n',origin)

"""
Basic One-time RAPPOR 
数据已经编码完成，并且只进行一次PRR
"""
epsilon=np.log(3) # eps越大，数据可用性越好，隐私保护水平越低
f=rpb.gen_f(epsilon)

z_totle=np.zeros(d)
for i in range(n):
    z=rpb.PRR_bits(onehot_data[i],f)
    z_totle=z_totle+z
print(z_totle)

est=np.zeros(d)
for j in range(d):
    est[j]=rpb.decoder_PRR(z_totle[j],n,f)
print('estimate:\n',est)


# 画图
plt.bar(x_list,+origin)
plt.bar(x_list,-est)
plt.show()

