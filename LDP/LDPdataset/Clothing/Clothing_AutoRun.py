"""
@author:FZX
@file:Movie_AutoRun.py
@time:2020/7/4 9:32
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.LDPKVbasic as lkvb
import numpy as np
import random
import time
import os

"""数据处理"""
# 关闭科学计数法显示
np.set_printoptions(suppress=True)

def LDP(n,l,k,total_eps,L):
    '''init'''
    # n=105571
    # l=5850
    # #topk的k
    # k=20
    #
    # L=2
    # # 总隐私预算
    # total_eps=10
    # 阶段一(eps1)和阶段二(eps2)隐私预算分割比率
    raito = 0.5
    eps1 = total_eps * raito
    eps2 = total_eps - eps1

    # eps1_1是阶段一查找候选集预算
    eps1_1 = 0.5 * eps1
    # eps1_2是阶段二上传交集项数的预算
    eps1_2 = eps1 - eps1_1

    eps2_1 = np.log((np.exp(eps2) + 1) / 2)
    eps2_2 = eps2
    a = 0.5
    p = lkvb.p_values(eps2_2)
    b = 1 - lkvb.p_values(eps2_1)
    print("total_eps:"+str(total_eps))

    # 数据路径定义
    root_data_path = "../Clothing/data/"
    # 结果路径定义
    root_result_path = "../Clothing/run/" + str(total_eps) + "/"
    if not os.path.exists(root_result_path):
        os.makedirs(root_result_path)
        print("路径" + root_result_path + "创建成功")

    k_path = root_data_path + "Clothing_k.txt"
    v_path = root_data_path + "Clothing_v.txt"
    samplek_path = root_data_path + "Clothing_samplek.txt"

    data_k = bf.readtxt(k_path)
    data_v = bf.readtxt(v_path)

    # 构建kv对元组列表，每一行代表一个用户的kv对list
    kv = []
    for i in range(n):
        data_k[i] = list(map(int, data_k[i]))
        tmp = zip(data_k[i], data_v[i])
        kv.append(list(tmp))

    '''阶段一：获取候选集'''
    print("阶段一：获取候选集")
    # 读取采样好的数据，每个用户那里采样一个
    # 注意！！！！要转换为和samplek相同的格式int类型，因为hash是专为str计算的，int和float转换得到的结果不同
    samplek = (bf.readtxt(samplek_path))[0]
    samplek = list(map(int, samplek))

    starttime = time.clock()
    print("开始执行OLH...")
    # 调用OLH
    est = lhb.OLH_1(eps1_1, samplek, n, l)
    # 程序运行结束时间
    endtime = time.clock()
    print('运行时间:', endtime - starttime)

    # 生成topk的候选集，候选集长度为2k
    est.sort(key=lambda x: x[1], reverse=True)
    topk_candi = est[:2 * k]
    # 在不会出现相同key的情况下，可以转为dict，提升速度
    candi_dict = dict(topk_candi)
    candi_k = list(candi_dict.keys())

    '''阶段一：获取填充长度'''
    print("阶段一：获取填充长度")
    # 这里用OLH获得的填充长度其实是固定的，就是juypter文件里计算得到的
    # 不需要再重复用OLH计算一遍了
    # L=2
    minik = []
    miniv = []
    minikv = []
    # 每个用户的交集项个数
    numsize = []
    for i in range(n):
        # data_k[i]是每个用户的项集的k值
        # 将data_k[i]与候选集candi_k取交集得到tmpk
        tmpk = list(set(candi_k).intersection(set(data_k[i])))
        # 根据tmpk得到对应项的v值tmpv
        di = dict(kv[i])
        tmpv = list(map(di.get, tmpk))
        minik.append(tmpk)
        miniv.append(tmpv)
        minikv.append(list(zip(tmpk, tmpv)))
        numsize.append(len(tmpk))

    '''阶段二：填充采样'''
    print("阶段二：填充采样")
    padlabel = [i + 1 for i in range(l, l + L)]
    cp_label = candi_k + padlabel
    upkv = [lkvb.P_S(minikv[i], l, L) for i in range(n)]

    '''阶段二：扰动聚合'''
    print("阶段二：扰动聚合")
    # 扰动
    d = 2 * k
    Y = lkvb.PCKV_UE(upkv, d, L, cp_label, a, p, b)
    # 聚合
    f_k, m_k = lkvb.AEC_UE(Y, d, L, n, a, p, b)



    # 最终结果，写入csv文件
    # 其实是一个查字典的过程，从真实的结果里面去查对应的count值
    # 以后要把这个文件从数据库里重新导出一份，要加上avg
    lable_path = root_data_path + 'Clothing_id_count.txt'
    id_count = bf.readtxt(lable_path)

    # 这里其他的要加上avg的真实和估计
    # 这里avg的真实数据暂时不在，以后再加
    # id_true_c_est_c_true_a_est_a简写成下面的把
    id_tcectaea = []

    for i in range(2 * k):
        index = int(candi_k[i])
        tmp = id_count[index - 1]
        id_tcectaea.append((index, tmp[0], tmp[1], n * f_k[i], 0, m_k[i]))

    # 获取时间作为文件名后缀
    s = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    filename = root_result_path + 'Clothing_result_' + str(total_eps) + '_' + s + '.csv'
    bf.savecsv(np.array(id_tcectaea), filename)


if __name__ == '__main__':
    LDP(n=105571,l=5850,k=20,total_eps=2,L=2)





