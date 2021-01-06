"""
@author:FZX
@file:PrivKVbasic.py
@time:2020/2/21 16:42
"""
import numpy as np
import random
import LDP.basicFunction.basicfunc as bf


def p_values(epsilon, n=2):
    """
    计算概率p的值
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    """
    p = np.e ** epsilon / (np.e ** epsilon + n - 1)
    return p


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


def correct(low, high, value):
    """
    矫正函数,value小于low，置为low，大于high置为high
    @param low:
    @param high:
    @param value:
    @return:
    """
    if value < low:
        value = low
    elif value > high:
        value = high
    return value


# def VPP(v, epsilon):
#     """
#     PriKV论文中的Algorithm2 VPP
#     :param v: kv对的v值,[-1,1]
#     :param epsilon: 隐私预算
#     :return:扰动后的v*
#     """
#     p_d = (1 + v) / 2  # 将v离散的概率
#     p_p = p_values(epsilon)  # 扰动v的概率
#
#     # 对v离散到v_d(1或-1)
#     rnd1 = np.random.random()
#     if rnd1 < p_d:
#         v_d = 1
#     else:
#         v_d = -1
#
#     # 将v_d扰动为v_p
#     rnd2 = np.random.random()
#     if rnd2 < p_p:
#         v_p = v_d
#     else:
#         v_p = -v_d
#
#     return v_p

def VPP(v, epsilon):
    """
    PriKV论文中的Algorithm2 VPP
    :param v: kv对的v值,[-1,1]
    :param epsilon: 隐私预算
    :return:扰动后的v*
    """
    p_d = (1 + v) / 2  # 将v离散的概率
    p_p = p_values(epsilon)  # 扰动v的概率

    # 对v离散到v_d(1或-1)
    v_d = mpp([1, -1], [p_d, 1 - p_d])

    # 将v_d扰动为v_p
    v_p = mpp([v_d, -v_d], [p_p, 1 - p_p])
    return v_p


# def LPP(k, v, epsilon1, epsilon2):
#     """
#     PriKV论文中的Algorithm3 LPP
#     :param k: kv对的k值
#     :param v: kv对的v值
#     :param epsilon1:
#     :param epsilon2:
#     :return: 扰动后的kv对
#     """
#     if k == 1:  # k=1表示这个这个k值存在于用户的项集S中
#         vp = VPP(v, epsilon2)  # 这里时论文中的v*
#         # 再对kv对进行扰动
#         rnd = np.random.random()
#         p = p_values(epsilon1)  # 扰动概率
#         if rnd < p:
#             k_p = 1  # 以大概率：k值不变，v值也不便
#             v_p = vp
#         else:
#             k_p = 0  # 以小概率：k和v都扰动为0
#             v_p = 0
#     else:  # k不为1，则表示这个k不在用户的项集S中
#         m = np.random.random() * 2 - 1  # 在[-1,1]区间内随机选择一个数作为一个虚拟值
#         vp = VPP(m, epsilon2)  # 得到这种情况下的v*
#         rnd = np.random.random()
#         p = p_values(epsilon1)  # 扰动概率
#         if rnd < p:
#             k_p = 0  # 以大概率：k和v值都扰动为0
#             v_p = 0
#         else:
#             k_p = 1  # 以小概率：k值扰动为1，v值不变
#             v_p = vp
#     return k_p, v_p

def LPP(k: list, v: list, d, epsilon1, epsilon2):
    """
    PriKV论文中的Algorithm3 LPP
    @param k: 用户的k list
    @param v: 用户的v list
    @param epsilon1:
    @param epsilon2:
    @param d: 数据维度
    @return:(k,v,j)元组
    """
    j = random.randint(1, d)  # 包括1和d
    p = p_values(epsilon1)  # 扰动概率
    rnd = np.random.random()
    if j in k:  # 表示这个这个k值存在于用户的项集S中
        index = k.index(j)
        vp = VPP(v[index], epsilon2)  # 这里时论文中的v*
        # 再对kv对进行扰动
        if rnd < p:
            k_p = 1  # 以大概率：k值不变，v值也不便
            v_p = vp
        else:
            k_p = 0  # 以小概率：k和v都扰动为0
            v_p = 0
    else:  # 则表示这个k不在用户的项集S中
        m = np.random.random() * 2 - 1  # 在[-1,1]区间内随机选择一个数作为一个虚拟值
        vp = VPP(m, epsilon2)  # 得到这种情况下的v*
        if rnd < p:
            k_p = 0  # 以大概率：k和v值都扰动为0
            v_p = 0
        else:
            k_p = 1  # 以小概率：k值扰动为1，v值不变
            v_p = vp
    return (k_p, v_p, j)


# def PrivKV(k, v, d, epsilon1, epsilon2):
#     """
#     PriKV论文中的Algorithm4 PrivKV
#     @param k: 所有用户的k集list
#     @param v: 所有用户的v集list
#     @param d: 数据维度
#     @param epsilon1:
#     @param epsilon2:
#     @return:
#     """
#
#     n = len(k)
#     all_kvp = [LPP(k[i], v[i], d, epsilon1, epsilon2) for i in range(n)]
#
#     total = 0
#     have = 0
#     pos = 0
#     neg = 0
#
#     for kv in all_kvp:
#         if kv[0] == 1:
#             have += 1
#         total += 1  # 这里的total不是全部用户的数量，而是每个k（j）的总数
#         if kv[1] == 1:
#             pos += 1
#         if kv[1] == -1:
#             neg += 1
#
#     f = have / total
#     p1 = p_values(epsilon1)
#     f = (p1 - 1 + f) / (2 * p1 - 1)
#     N = pos + neg
#     p2 = p_values(epsilon2)
#     n1 = (p2 - 1) / (2 * p2 - 1) * n + pos / (2 * p2 - 1)
#     n2 = (p2 - 1) / (2 * p2 - 1) * n + neg / (2 * p2 - 1)
#
#     if n1 < 0:
#         n1 = 0
#     elif n1 > N:
#         n1 = N
#
#     if n2 < 0:
#         n2 = 0
#     elif n2 > N:
#         n2 = N
#
#     m = (n1 - n2) / N
#     return f, m


def PrivKV(k, v, d, epsilon1, epsilon2):
    """
    PriKV论文中的Algorithm4 PrivKV
    @param k: 所有用户的k集list
    @param v: 所有用户的v集list
    @param d: 数据维度
    @param epsilon1:
    @param epsilon2:
    @return:
    """

    n = len(k)
    all_kvp = [LPP(k[i], v[i], d, epsilon1, epsilon2) for i in range(n)]
    p1 = p_values(epsilon1)
    p2 = p_values(epsilon2)
    pos = [0 for i in range(d)]
    neg = [0 for i in range(d)]
    count = [0 for i in range(d)]
    for kv in all_kvp:
        if kv[0] == 0:
            continue
        j = kv[2]
        if (kv[1] == 1):
            pos[j - 1] += 1
            count[j - 1] += 1
        if (kv[1] == -1):
            neg[j - 1] += 1
            count[j - 1] += 1
    f = np.array(count) / n  # 频率
    f_k = ((p1 - 1 + f) / (2 * p1 - 1)).tolist()

    pos = np.array(pos)
    neg = np.array(neg)
    N = pos + neg
    n1 = N * (p2 - 1) / (2 * p2 - 1) + pos / (2 * p2 - 1)
    n2 = N * (p2 - 1) / (2 * p2 - 1) + neg / (2 * p2 - 1)

    for i in range(d):
        n1[i] = correct(0, N[i], n1[i])
        n2[i] = correct(0, N[i], n2[i])
        if(N[i]==0):
            N[i]=1
    m_k = ((n1 - n2) / N).tolist()

    return f_k, m_k


def gettrue(k, v):
    """
    计算真实的f和m值
    :param k:
    :param v:
    :return:
    """
    total = 0
    have = 0
    value = 0

    k_v = list(zip(k, v))
    for d in k_v:
        if d[0] == 1:
            have += 1
            value += d[1]
        total += 1
    f_true = have / total
    m_true = value / have

    return f_true, m_true


if __name__ == '__main__':
    k = [2, 1, 3, 4, 5]
    v = [0.4, 0.1, -0.2, 0.5, -0.1]
    r = LPP(k, v, 5, 0.4, 0.5)
    print(r)
