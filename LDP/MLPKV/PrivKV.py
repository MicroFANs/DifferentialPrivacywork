"""
@author:FZX
@file:PrivKV.py
@time:2021/1/6 3:11 下午
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.LDPKVbasic as lkvb
import LDP.basicDP.PCKVbasic as pcb
import LDP.basicDP.MLPKVbasic as mlkvp
import LDP.basicDP.PrivKVbasic as prkvb
import numpy as np
import random
import time


"""数据处理"""
# 关闭科学计数法显示
np.set_printoptions(suppress=True)

def PrivKV(param):
    data_name = param[0]  # 数据名
    epsilon = param[1]  # 隐私预算
    k = param[2]  # top_k的前k项

    epsilon1=epsilon/2
    epsilon2=epsilon/2

    k_path='../MLPKV/data/'+data_name+'/'+data_name+'_k.txt'
    v_path='../MLPKV/data/'+data_name+'/'+data_name+'_v.txt'
    true_path='../MLPKV/data/'+data_name+'/'+data_name+'_id_count_avg_norm.txt'

    data_k = bf.readtxt(k_path)
    data_v = bf.readtxt(v_path)
    true_count= bf.readtxt(true_path)

    n = len(data_k)  # 用户数量
    d = len(true_count)  # key域的长度
    # 构建kv对元组列表，每一行代表一个用户的kv对list
    kv = []

    for i in range(n):
        data_k[i]=list(map(int,data_k[i]))
        tmp = zip(data_k[i], data_v[i])
        kv.append(list(tmp))

    # 构建label标签，我已经把标签规范化为[1,l]的数字了，也就是说不需要用之前的原始标签了,不用查字典了
    label=[i+1 for i in range(d)]

    f_k, m_k = prkvb.PrivKV(data_k,data_v,d,epsilon1,epsilon2)

    est = []  # 估计结果按k频率排序
    for i in range(d):
        est.append((i+1, f_k[i], m_k[i]))
    est.sort(key=lambda x: x[1], reverse=True)

    est_f = {}  # 频率估计值字典
    est_m = {}  # 均值估计值字典
    Candidate_rank = []  # 按照频率排序的候选集id
    for i in range(2 * k):
        est_f[est[i][0]] = est[i][1]
        est_m[est[i][0]] = est[i][2]
        Candidate_rank.append(est[i][0])


    true_f = np.load('../MLPKV/data/' + data_name + '/' + data_name + '_true_f.npy').item()
    true_m = np.load('../MLPKV/data/' + data_name + '/' + data_name + '_true_m.npy').item()
    true_rank = bf.readtxt('../MLPKV/data/' + data_name + '/' + data_name + '_true_rank.txt')[0]

    ## 指标
    ## 1.命中率
    hit_ratio = mlkvp.hit_ratio(Candidate_rank, true_rank, k)

    ## 2.MSE_f
    MSE_f = mlkvp.MSE(Candidate_rank, est_f, true_f, k)

    ## 3.MSE_m
    MSE_m = mlkvp.MSE(Candidate_rank, est_m, true_m, k)

    # 保存结果
    save_path = '../MLPKV/result/PrivKV_result.txt'
    time_stamp = time.strftime("%Y-%m-%d %X")
    result = [[data_name, epsilon, k, hit_ratio, MSE_f, MSE_m, time_stamp]]
    bf.savetxt(result, save_path)
    return 0

def get_parameters():
    eps = [6, 5, 4, 3, 2, 1, 0.5, 0.1]
    data_names = ['Clothing', 'Ecomm', 'Movie', 'Amazon']
    s_data=['Uniform','Uniform_d','Gaussian','Gaussian_d']
    data_padding = {'Ecomm': 1, 'Clothing': 2, 'Movie': 25, 'Amazon': 2}
    parameters = []
    for name in s_data:
        for e in eps:
            parameters.append([name, e, 20])
    return parameters

if __name__ == '__main__':
    # parameters=[data_name,epsilon,padding_length,top_k]
    parameters = get_parameters()
    print(parameters)
    for param in parameters:
        for i in range(5):
            print("执行，data：", param[0], "eps:", param[1], "第", i, "次")
            PrivKV(param)