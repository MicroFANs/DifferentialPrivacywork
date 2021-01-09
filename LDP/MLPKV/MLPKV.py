"""
@author:FZX
@file:MLPKV.py
@time:2021/1/4 9:24 下午
"""
import matplotlib
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.LDPKVbasic as lkvb
import LDP.basicDP.MLPKVbasic as mlkvp
import numpy as np
import random
import time


def MPLKV(param):
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
    print('运行时间:', endtime1 - starttime1)

    est_A_temp = list(zip(label, est_A))
    est_A_temp.sort(key=lambda x: x[1], reverse=True)

    # 候选集
    candi_dict = dict(est_A_temp[:2 * k])
    Candidate = list(candi_dict.keys())

    ## GroupB用户完成Step3和4
    ## Step3：取交集并扰动上传
    ## Step4：获得填充长度返回

    ## GroupC用户完成Step5和6
    ## Step5：填充采样并上传
    ## (1)GroupC也和Candidate取交集

    GroupC_inter = []  # GroupC组用户和Candidate的交集
    GroupC_k = []  # GroupC组用户拥有的k值
    GroupC_v = []  # GroupC组用户拥有的v值
    for i in range(n_C):
        each_k = list(zip(*GroupC[i]))[0]
        temp_k = set(Candidate).intersection(set(each_k))
        GroupC_dict = dict(GroupC[i])
        temp_v = list(map(GroupC_dict.get, temp_k))
        GroupC_inter.append(list(zip(temp_k, temp_v)))

    ## (2)Padding and Sampling
    pad_item = [i + 1 for i in range(d, d + padding_length)]
    pad_label = Candidate + pad_item
    PS_C = [mlkvp.P_S(GroupC_inter[i], d, padding_length) for i in range(n_C)]

    ## (3)UE扰动
    eps1 = np.log((np.exp(epsilon) + 1) / 2)
    eps2 = epsilon
    a = 0.5
    p = lkvb.p_values(eps2)
    b = 1 - lkvb.p_values(eps1)
    perturbed_C = mlkvp.PCKV_UE(PS_C, 2 * k, padding_length, pad_label, a, p, b)

    ## Step6：估计频率和均值
    f_k, m_k = mlkvp.AEC_UE(perturbed_C, 2 * k, padding_length, n_C, a, p, b)
    # print(f_k)
    # print(m_k)

    est = []  # 估计结果按k频率排序
    for i in range(2 * k):
        est.append((Candidate[i], f_k[i], m_k[i]))
    est.sort(key=lambda x: x[1], reverse=True)

    est_f = {}  # 频率估计值字典
    est_m = {}  # 均值估计值字典
    Candidate_rank = []  # 按照频率排序的候选集id
    for i in range(2 * k):
        est_f[est[i][0]] = est[i][1]
        est_m[est[i][0]] = est[i][2]
        Candidate_rank.append(est[i][0])
    # print('est_f:', est_f)
    # print('est_m:', est_m)
    # print("Candidate_rank:", Candidate_rank)

    ##### 获取实际结果
    true_f = {}  # 真实频率字典
    true_k = {}  # 真实的项目（不用除以n）
    true_m = {}  # 真实均值字典
    true_data = []  # 真实频率排名元组list(id,k_count,v_mean)
    true_rank = []  # 真实的频率排名
    # for i in range(d):
    #     true_k[i + 1] = true_count[i][1]
    #     true_f[i + 1] = true_count[i][1] / n
    #     true_m[i + 1] = true_count[i][2]
    #     true_data.append((i + 1, true_count[i][1], true_count[i][2]))
    #
    # true_data.sort(key=lambda x: x[1], reverse=True)
    # for i in range(d):
    #     true_rank.append(true_data[i][0])
    # # print("true_data:", true_data)
    # # print("true_f:", true_f)
    # # print("true_k:", true_k)
    # # print("true_m:", true_m)
    # # print("true_rank:", true_rank)
    # 直接读取文件
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
    save_path = '../MLPKV/result/result.txt'
    time_stamp = time.strftime("%Y-%m-%d %X")
    result = [[data_name, epsilon, padding_length, k, hit_ratio, MSE_f, MSE_m,time_stamp]]
    bf.savetxt(result, save_path)
    return 0


def Group_hit(param):
    data_name = param[0]  # 数据名
    epsilon = param[1]  # 隐私预算
    padding_length = param[2]  # 填充长度
    k = param[3]  # top_k的前k项
    G1 = param[4]
    G2 = param[5]
    G3 = param[6]

    '''路径名'''
    k_path = '../MLPKV/data/' + data_name + '/' + data_name + '_k.txt'
    v_path = '../MLPKV/data/' + data_name + '/' + data_name + '_v.txt'
    true_path = '../MLPKV/data/' + data_name + '/' + data_name + '_id_count_avg_norm.txt'

    data_k = bf.readtxt(k_path)
    data_v = bf.readtxt(v_path)
    true_count = bf.readtxt(true_path)

    n = len(data_k)  # 用户数量
    d = len(true_count)  # key域的长度

    # 构建kv对元组列表，每一行代表一个用户的kv对list
    kv = []
    for i in range(n):
        data_k[i] = list(map(int, data_k[i]))
        tmp = zip(data_k[i], data_v[i])
        kv.append(list(tmp))

    # 构建label标签，我已经把标签规范化为[1,l]的数字了，也就是说不需要用之前的原始标签了,不用查字典了
    label = [i + 1 for i in range(d)]

    # 分组，GroupA完成候选项集，GroupB完成填充项集，GroupC完成估计
    GroupA, GroupB, GroupC = mlkvp.divide_list(kv, G1, G2, G3)
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
    est_A = lhb.OLH_aggregation_mutilprocess(epsilon, sampling_perturbed_A, n_A, d, lhb.get_support, 7)
    # 程序运行结束时间
    endtime1 = time.time()
    print('运行时间:', endtime1 - starttime1)

    est_A_temp = list(zip(label, est_A))
    est_A_temp.sort(key=lambda x: x[1], reverse=True)

    # 候选集
    candi_dict = dict(est_A_temp[:2 * k])
    Candidate = list(candi_dict.keys())

    GroupC_inter = []  # GroupC组用户和Candidate的交集
    GroupC_k = []  # GroupC组用户拥有的k值
    GroupC_v = []  # GroupC组用户拥有的v值
    for i in range(n_C):
        each_k = list(zip(*GroupC[i]))[0]
        temp_k = set(Candidate).intersection(set(each_k))
        GroupC_dict = dict(GroupC[i])
        temp_v = list(map(GroupC_dict.get, temp_k))
        GroupC_inter.append(list(zip(temp_k, temp_v)))

    ## (2)Padding and Sampling
    pad_item = [i + 1 for i in range(d, d + padding_length)]
    pad_label = Candidate + pad_item
    PS_C = [mlkvp.P_S(GroupC_inter[i], d, padding_length) for i in range(n_C)]

    ## (3)UE扰动
    eps1 = np.log((np.exp(epsilon) + 1) / 2)
    eps2 = epsilon
    a = 0.5
    p = lkvb.p_values(eps2)
    b = 1 - lkvb.p_values(eps1)
    perturbed_C = mlkvp.PCKV_UE(PS_C, 2 * k, padding_length, pad_label, a, p, b)

    ## Step6：估计频率和均值
    f_k, m_k = mlkvp.AEC_UE(perturbed_C, 2 * k, padding_length, n_C, a, p, b)

    est = []  # 估计结果按k频率排序
    for i in range(2 * k):
        est.append((Candidate[i], f_k[i], m_k[i]))
    est.sort(key=lambda x: x[1], reverse=True)
    Candidate_rank = []  # 按照频率排序的候选集id
    for i in range(2 * k):
        Candidate_rank.append(est[i][0])

    ##### 获取实际结果
    # print(true_count)
    true_data = []  # 真实频率排名元组list(id,k_count,v_mean)
    true_rank = []  # 真实的频率排名
    for i in range(d):
        true_data.append((i + 1, true_count[i][1], true_count[i][2]))
    true_data.sort(key=lambda x: x[1], reverse=True)
    for i in range(d):
        true_rank.append(true_data[i][0])

    ## 指标
    ## 1.命中率
    hit_ratio = mlkvp.hit_ratio(Candidate_rank, true_rank, k)

    # 保存结果
    save_path = '../MLPKV/result/Group_hit.txt'
    time_stamp = time.strftime("%Y-%m-%d %X")
    result = [[data_name, epsilon, padding_length, k, hit_ratio, str(G1) + ':' + str(G2) + ':' + str(G3), time_stamp]]
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
            parameters.append([name, e, 1, 20])
    return parameters


if __name__ == '__main__':
    # parameters=[data_name,epsilon,padding_length,top_k]
    # param = ['Clothing', 5, 2, 20]
    # MPLKV(param)
    # # parameters=[['Clothing', 6, 2, 20], ['Clothing', 5, 2, 20],['Ecomm', 6, 1, 20], ['Ecomm', 5, 1, 20]]

    parameters = get_parameters()
    print(parameters)
    for param in parameters:
        for i in range(5):
            print("执行，data：", param[0], "eps:", param[1], "第", i, "次")
            MPLKV(param)

