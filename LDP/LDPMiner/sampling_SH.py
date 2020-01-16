"""
@author:FZX
@file:sampling_SH.py
@time:2019/11/25 16:40
"""

"""
@author:FZX
@file:myLDPAlgorithm.py
@time:2019/12/30 4:12 下午
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


# 读取数据
path='../LDPMiner/dataset/kosarak/kosarak.txt'
user_data=bf.readtxt(path)

# all_iterm是存放所有出现过的项的数组，用来查找项对应的index，以便进行编码
iterm_list=bf.combine_lists(user_data)
label=np.unique(np.array(iterm_list))
x_list=label.tolist() # 数据标签
print(len(x_list))




# d=len(label) # 获取数据维度
# n=len(user_data) # 获取用户数

#kosarak.txt数据集的参数
d=41270
n=990002
l=20 # 设置的项集长度为l



#sampling过程，构造私有项集
# 别保存在github仓库中，超过100M不让上传


#savepath='../LDPMiner/dataset/kosarak/l_set.txt'
v_list=bf.gen_iterm_set(user_data,l)
#bf.savetxt(v_list,savepath)
#print(v_list[0])

sampling_data=np.zeros(n)
for i in range(n):
    sampling_data[i]=random.choice(v_list[i])
print(type(sampling_data))

# 释放内存
del v_list
#bf.savetxt(sampling_data,'../LDPMiner/dataset/kosarak/sampling_data.txt')

"""
将原来990002个用户分组，每组包含1000个用户
"""


epsilon=np.log(3)

# 计算随机投影矩阵参数
r,m=shb.comput_parameter(d=d,n=1000,epsilon=epsilon,beta=0.01)
print('m=',m)
# 生成随机投影矩阵，这里的虚拟项0我也将其当作一个项到最后不计算它的频率
rnd_proj=shb.genProj(m,d)


#server
z_total=np.zeros(m)
for i in range(990002):
    x=int(sampling_data[i])
    #print(x)
    z=shb.Basic_Randomizer_1(m=m,x=x,epsilon=epsilon,rand_proj=rnd_proj)
    #print(z)
    z_total=z_total+z
z_mean=z_total/990002
print(z_mean)


v=1

f_est=shb.FO(rnd_proj,z_mean,v)
print('v=',v,'\nf_estmiate=',f_est)

# 评估d