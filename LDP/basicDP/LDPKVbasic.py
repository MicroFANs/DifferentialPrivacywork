"""
@author:FZX
@file:LDPKVbasic.py
@time:2020/5/22 9:34 上午
"""
import numpy as np
import random


def p_values(epsilon, n=2):
    """
    计算概率p的值
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    """
    p = np.e ** epsilon / (np.e ** epsilon + n - 1)
    return p


def P_S(k_v, d: int, l: int):
    """
    PCKV论文中的Algorithm1 Padding and Sampling
    以高概率b从用户的项集S中采样，挑选一个kv对
    以低概率（1-b）从扩充的虚拟项集l中挑选虚拟k值，并且v值值0
    此时总的k值的维度也从d变成了(d+l)
    @param k_v: 用户kv值元组[(1,0.4),(2,-0.2),(3,0.5)]
    @param d: key的维度
    @param l: 填充长度L
    @return: 采样元组
    """
    S = len(k_v)
    b = S / (max(S, l))
    rnd = np.random.random()
    if rnd < b:  # 从用户项集S中随机选择一个kv对，从[0,S)中随机选择一个数作为序号
        tmp = random.sample(k_v, 1)
        k_ps = tmp[0][0]
        v_ps = tmp[0][1]
    else:
        v_ps = 0
        k_ps = np.random.randint(d, d + l) + 1  # 从{d+1,d+2,...,d'}中随机选择一个作为序号

    # 离散v_ps值
    p = (1 + v_ps) / 2
    rnd = np.random.random()
    if rnd < p:
        v_ps = 1
    else:
        v_ps = -1

    return k_ps, v_ps


def mpp(candidate: list, p: list):
    """
    多概率扰动函数 mutil_prob_perturbation
    以不同的概率选择不同的元素,如以0.2的概率选择1，以0.3的概率选择-1，以0.5的概率选择0
    就写成mpp([1,-1,0],[0.2,0.3,0.5]),概率和所选值要对应
    :param candidate:不同候选元素集合 list
    :param p: 不同概率集合 list
    :return:输出的值
    """
    v = np.random.choice(candidate, p=p)
    return v


def P_UE(k_v, d, l, label, a, p, b):
    """
    PCKV中相应算法的改进
    @param k_v: 每个用户采样得到的元组(3,0.4)
    @param d: 这里的维度与PCKV论文中的不同，这里是candidate的维度，就是2k
    @param l: 填充项长度L
    @param label: 就是candidate+padlable(长度是d+l=2k+L)
    @param a: 真实k保持不变的概率
    @param p: 真实v保持不变的概率
    @param b: 虚假k反转为1的概率
    @return:UE list[1,0,1,0,...,0]
    """
    y = [mpp([1, -1, 0], [b / 2, b / 2, (1 - b)]) for i in range(d + l)]
    k = k_v[0]
    v = k_v[1]
    indx = label.index(k)
    y[indx] = mpp([v, -v, 0], [a * p, a * (1 - p), (1 - a)])
    return y


def PCKV_UE(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法的改进
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d: 这里的维度与PCKV论文中的不同，这里是candidate的维度，就是2k
    @param l: 填充项长度L
    @param label: 就是candidate
    @param a: 真实k保持不变的概率
    @param p: 真实v保持不变的概率
    @param b: 虚假k反转为1的概率
    @return: 嵌套表list
    """
    Y = [P_UE(x, d, l, label, a, p, b) for x in all_kv]
    return Y


# def AEC_UE(kv_p,d,l,label,a,p,b):


#
def P_GRR(k_v, d, l, label, a, p):
    """
    PCKV中相应算法改进
    @param k_v:k_v对
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate
    @param a:
    @param p:
    @return:
    """
    k = k_v[0]
    v = k_v[1]
    b = (1 - a) / (d + l - 1)

    indk = label.index(k)
    indx = indk
    rnd = np.random.random_sample()
    if rnd > a:
        while indx == indk:
            indx = np.random.randint(0, (d + l))
        k_p = label[indx]
        v_p = mpp([1, -1], [0.5, 0.5])
    else:
        k_p = k
        v_p = mpp([v, -v], [p, (1 - p)])
    return k_p, v_p


def PCKV_GRR(all_kv, d, l, label, a, p):
    """
    PCKV中相应算法
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate
    @param a:
    @param p:
    @return:
    """
    Y = [PCKV_GRR(x, d, l, label, a, p) for x in all_kv]
    return Y
