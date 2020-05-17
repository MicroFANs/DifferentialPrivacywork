"""
@author:FZX
@file:LDPKV.py
@time:2020/5/12 8:41 下午
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

k_path = '../LDPdataset/Clothing/data/Clothing_k.txt'
v_path = '../LDPdataset/Clothing/data/Clothing_v.txt'
lable_path = '../LDPdataset/Clothing/data/Clothing_lable.txt'

data_k = bf.readtxt(k_path)
data_v = bf.readtxt(v_path)
data_l = bf.readtxt(lable_path)


n = len(data_k)  # 用户数量
l = len(data_l[0])  # key域的长度
epsilon = 5

# 构建kv对元组列表，每一行代表一个用户的kv对list
kv = []
for i in range(n):
    tmp = zip(data_k[i], data_v[i])
    kv.append(list(tmp))

"""采样"""
# 通过采样（每个用户随机抽取一个kv对），构建上传upkvlist，长度为用户数目n，表示每个用户上传对kv对
upkv = []
for i in range(n):
    random.seed(1)
    tmp = random.sample(kv[i], 1)
    upkv.append(tmp[0])
#print(upkv)
upkv_unzip = list(zip(*upkv))
upk = upkv_unzip[0]
upv = upkv_unzip[1]
upk = list(map(int, upk))

label = list(map(int,data_l[0]))  # 查找数据用的label
# 注意！！！！要转换为和upk相同的格式，因为hash是专为str计算的，int和float
#转换得到的结果不同



count_true = []  # 真实计数结果
for i in range(l):
    count_true.append((label[i],upk.count(label[i])))
print(count_true)



"""用OLH查找候选集"""
# 程序运行起始时间
starttime = time.clock()
print("开始执行OLH...")

# 调用OLH
est = lhb.OLH(epsilon, upk, n, label)
print(est)





# 程序运行结束时间
endtime = time.clock()
print('运行时间:', endtime - starttime)
#
# savepath = '../LDPdataset/Clothing/result/candidate.txt'
# sp='../LDPdataset/Clothing/result/true.txt'
#upkpath='../LDPdataset/Clothing/result/upk.txt'


#
# bf.savetxt(est, savepath)
# bf.savetxt(count_true,sp)
#bf.savetxt(upk,upkpath)



# #单个值测试
# g=int(np.exp(epsilon))+1
# starttime=time.clock()
# x=[]
# for i in range(n):
#     value=upk[i]
#     index=lhb.olh_hash(value,g,i)
#     x.append(index)
# print(x)
#
# p=lhb.p_values(epsilon,n=g)
# y=[]
# for i in range(n):
#     y.append(lhb.olh_grr(p,x[i],g))
# print(y)
#
# #v=123373
# v=label[0]
# print(v)
# c=lhb.olh_support(v,y,g,n)
# print(c)
# e=lhb.aggregation(c,n,g,p)
# print(e)