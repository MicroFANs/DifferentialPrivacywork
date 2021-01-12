"""
@author:FZX
@file:EC_OLH.py
@time:2021/1/12 3:53 下午
"""

#比较EC-OLH和OLH的运行效率
import matplotlib
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.LDPKVbasic as lkvb
import LDP.basicDP.MLPKVbasic as mlkvp
import numpy as np
import random
import time


def ECOLH(param):
    data_name = param[0]  # 数据名
    epsilon = param[1]  # 隐私预算
    padding_length = param[2]  # 填充长度
    k = param[3]  # top_k的前k项

    '''路径名'''
    k_path = '../MLPKV/data/' + data_name + '/' + data_name + '_k.txt'
    v_path = '../MLPKV/data/' + data_name + '/' + data_name + '_v.txt'
    true_path = '../MLPKV/data/' + data_name + '/' + data_name + '_id_count_avg_norm.txt'
    true_f_path = '../MLPKV/data/' + data_name + '/' + data_name + '_true_f.npy'
    true_m_path = '../MLPKV/data/' + data_name + '/' + data_name + '_true_m.npy'
    true_rank_path = '../MLPKV/data/' + data_name + '/' + data_name + '_true_rank.txt'

    data_k = bf.readtxt(k_path)
    data_v = bf.readtxt(v_path)
    true_count = bf.readtxt(true_path)

    n = len(data_k)  # 用户数量
    d = len(true_count)  # key域的长度

    # 构建kv对元组列表，每一行代表一个用户的kv对list`
    kv = []
    for i in range(n):
        data_k[i] = list(map(int, data_k[i]))
        tmp = zip(data_k[i], data_v[i])
        kv.append(list(tmp))

    # 构建label标签，我已经把标签规范化为[1,l]的数字了，也就是说不需要用之前的原始标签了,不用查字典了
    label = [i + 1 for i in range(d)]

    # 分组，GroupA完成候选项集，GroupB完成填充项集，GroupC完成估计
    GroupA, GroupB, GroupC = mlkvp.divide_list(kv, 4, 2, 4)
    n_A = len(GroupA)  # A组用户数量
    n_B = len(GroupB)  # B组用户数量
    n_C = len(GroupC)  # C组用户数量

    ## GroupA用户完成Step1和2
    # Step1：采样并扰动上传

    # 从每个用户项集里随机采样一个key并扰动
    sampling_perturbed_A = []
    for i in range(n_A):
        random.seed(1)
        sampling_key = random.sample(GroupA[i], 1)[0][0]  # 随机采样得到的项
        perturbed_key = lhb.OLH_perturbation(epsilon, sampling_key, i)
        sampling_perturbed_A.append((perturbed_key, i))

    ## Step2：聚合得到候选集

    starttime1 = time.time()
    print("开始执行OLH...")
    # 注意这里的用户数目n要用组A的用户数目，而不是所有的
    est_A = lhb.OLH_aggregation_mutilprocess(epsilon, sampling_perturbed_A, n_A, d, lhb.get_support, 8)
    # 程序运行结束时间
    endtime1 = time.time()
    t1=endtime1-starttime1
    print('运行时间:', t1)

    #用做对比时间，这是没有用多线程的，上面是用了多进程的EC-OLH

    starttime2 = time.time()
    print("开始执行OLH...")
    support=[]
    g = int(np.exp(epsilon)) + 1  # 参数g
    for i in range(d):
        c=0
        for j in range(len(sampling_perturbed_A)):
            if(lhb.olh_hash(i+1,g,sampling_perturbed_A[j][1])==sampling_perturbed_A[j][0]):
                c+=1
        support.append(c)


    e=[]
    p = lhb.p_values(epsilon, n=g)  # 参数p

    for i in range(len(support)):
        es=lhb.aggregation(support[i],len(GroupA),g,p)
        e.append(es)
    endtime2 = time.time()
    t2=endtime2-starttime2
    print('运行时间:', t2)

    # 保存结果
    save_path = '../MLPKV/result/EC_OLH_time.txt'
    time_stamp = time.strftime("%Y-%m-%d %X")
    result = [[data_name,t1,t2,time_stamp]]
    bf.savetxt(result, save_path)

    return 0

def get_parameters():
    eps = [6, 5, 4, 3, 2, 1, 0.5, 0.1]
    data_names = ['Clothing', 'Ecomm', 'Movie']
    s_data=['Uniform','Gaussian']
    data_padding = {'Ecomm': 1, 'Clothing': 2, 'Movie': 25, 'Amazon': 2}
    parameters = []
    for name in data_names:
        parameters.append([name,5,data_padding[name],20])
    for name in s_data:
        parameters.append([name, 5, 1, 20])

    return parameters

if __name__ == '__main__':
    # parameters=[data_name,epsilon,padding_length,top_k]
    # param = ['Clothing', 5, 2, 20]
    # MPLKV(param)
    # # parameters=[['Clothing', 6, 2, 20], ['Clothing', 5, 2, 20],['Ecomm', 6, 1, 20], ['Ecomm', 5, 1, 20]]

    parameters = get_parameters()
    print(parameters)
    for param in parameters:
        print("执行，data：", param[0], "eps:", param[1])
        ECOLH(param)