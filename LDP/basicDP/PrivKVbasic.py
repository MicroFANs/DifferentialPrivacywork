"""
@author:FZX
@file:PrivKVbasic.py
@time:2020/2/21 16:42
"""
import numpy as np
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
    rnd1 = np.random.random()
    if rnd1 < p_d:
        v_d = 1
    else:
        v_d = -1

    # 将v_d扰动为v_p
    rnd2 = np.random.random()
    if rnd2 < p_p:
        v_p = v_d
    else:
        v_p = -v_d

    return v_p


def LPP(k, v, epsilon1, epsilon2):
    """
    PriKV论文中的Algorithm3 LPP
    :param k: kv对的k值
    :param v: kv对的v值
    :param epsilon1:
    :param epsilon2:
    :return: 扰动后的kv对
    """
    if k == 1:  # k=1表示这个这个k值存在于用户的项集S中
        vp = VPP(v, epsilon2)  # 这里时论文中的v*
        # 再对kv对进行扰动
        rnd = np.random.random()
        p = p_values(epsilon1)  # 扰动概率
        if rnd < p:
            k_p = 1  # 以大概率：k值不变，v值也不便
            v_p = vp
        else:
            k_p = 0  # 以小概率：k和v都扰动为0
            v_p = 0
    else:  # k不为1，则表示这个k不在用户的项集S中
        m = np.random.random() * 2 - 1  # 在[-1,1]区间内随机选择一个数作为一个虚拟值
        vp = VPP(m, epsilon2)  # 得到这种情况下的v*
        rnd = np.random.random()
        p = p_values(epsilon1)  # 扰动概率
        if rnd < p:
            k_p = 0  # 以大概率：k和v值都扰动为0
            v_p = 0
        else:
            k_p = 1  # 以小概率：k值扰动为1，v值不变
            v_p = vp
    return k_p, v_p


def PrivKV(k, v, epsilon1, epsilon2):
    """
    PriKV论文中的Algorithm4 PrivKV
    这里的参数k和v并不是直接输入从txt中读取的原始数据
    因为LPP函数中少了一步采样，因此需要先对数据进行采样
    再保存到k和v的list中去，也就是说k中的数据是1或者0，
    表示在不在用户项集S中，v中的元素就是0或者本来的v值

    还有一点，就是这个实现的算法是默认指定一个k的，在算法第4行
    对K中的每个k来执行这个算法，实际上还需要一步就是将LPP之后每个用户
    提交的结果（序号j，和扰动kv值）按照j来分组保存，然后对于每一个j执行
    算法的5-12行才能得到结果
    :param k:k值的list
    :param v:v值的list
    :param epsilon1:
    :param epsilon2:
    :return:
    """
    n = len(k)
    all_kvp = [LPP(k[i], v[i], epsilon1, epsilon2) for i in range(n)]

    total=0
    have = 0
    pos = 0
    neg = 0

    for kv in all_kvp:
        if kv[0] == 1:
            have += 1
        total+=1 # 这里的total不是全部用户的数量，而是每个k（j）的总数
        if kv[1] == 1:
            pos += 1
        if kv[1] == -1:
            neg += 1

    f = have / total
    p1 = p_values(epsilon1)
    f = (p1 - 1 + f) / (2 * p1 - 1)
    N = pos + neg
    p2 = p_values(epsilon2)
    n1 = (p2 - 1) / (2 * p2 - 1) * n + pos / (2 * p2 - 1)
    n2 = (p2 - 1) / (2 * p2 - 1) * n + neg / (2 * p2 - 1)

    if n1<0:
        n1=0
    elif n1>N:
        n1=N

    if n2<0:
        n2=0
    elif n2>N:
        n2=N

    m=(n1-n2)/N
    return f,m


def gettrue(k,v):
    """
    计算真实的f和m值
    :param k:
    :param v:
    :return:
    """
    total=0
    have=0
    value=0

    k_v=list(zip(k,v))
    for d in k_v:
        if d[0]==1:
            have+=1
            value+=d[1]
        total+=1
    f_true=have/total
    m_true=value/have

    return f_true,m_true
