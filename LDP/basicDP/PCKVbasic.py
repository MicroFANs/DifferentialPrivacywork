"""
@author:FZX
@file:PCKVbasic.py
@time:2020/2/29 18:33
"""
import numpy as np
import random
from multiprocessing import Pool

#
# def p_values(epsilon, n=2):
#     """
#     计算概率p的值
#     :param epsilon: 隐私预算
#     :param n: 默认n=2是二元的，也可以是多元的
#     :return:概率p的值
#     """
#     p = np.e ** epsilon / (np.e ** epsilon + n - 1)
#     return p
#
#
# def PS(k_v, d:int, l:int):
#     """
#     PCKV论文中的Algorithm1 Padding and Sampling
#     以高概率b从用户的项集S中采样，挑选一个kv对
#     以低概率（1-b）从扩充的虚拟项集l中挑选虚拟k值，并且v值值0
#     此时总的k值的维度也从d变成了(d+l)
#     @param k_v: 用户kv值元组[(1,0.4),(2,-0.2),(3,0.5)]
#     @param d: key的维度
#     @param l: 填充长度L
#     @return: 采样元组
#     """
#     S = len(k_v)
#     b = S / (max(S, l))
#     rnd = np.random.random()
#     if rnd < b:  # 从用户项集S中随机选择一个kv对，从[0,S)中随机选择一个数作为序号
#         tmp=random.sample(k_v,1)
#         k_ps =tmp[0][0]
#         v_ps = tmp[0][1]
#     else:
#         v_ps = 0
#         k_ps = np.random.randint(d, d+l) + 1  # 从{d+1,d+2,...,d'}中随机选择一个作为序号
#
#     # 离散v_ps值
#     p = (1 + v_ps) / 2
#     rnd = np.random.random()
#     if rnd < p:
#         v_ps = 1
#     else:
#         v_ps = -1
#
#     return k_ps, v_ps
#
#
#
#
#
#
# def mpp(candidate:list, p:list):
#     """
#     多概率扰动函数 mutil_prob_perturbation
#     以不同的概率选择不同的元素,如以0.2的概率选择1，以0.3的概率选择-1，以0.5的概率选择0
#     就写成mpp([1,-1,0],[0.2,0.3,0.5]),概率和所选值要对应
#     :param candidate:不同候选元素集合 list
#     :param p: 不同概率集合 list
#     :return:输出的值
#     """
#     v = np.random.choice(candidate, p=p)
#     return v
#
#
# def P_UE(k_v, d, l, a, p, b):
#     """
#     PCKV_UE算法中的扰动部分
#     :param k_v: k_v是元组,(k,v),如(3,0.4)
#     :param d:
#     :param l:
#     :param a:
#     :param p:
#     :param b:
#     :return:
#     """
#     y = [mpp([1, -1, 0], [b / 2, b / 2, (1 - b)]) for i in range(d + l)]
#     k = k_v[0]
#     v = k_v[1]
#     y[k - 1] = mpp([v, -v, 0], [a * p, a * (1 - p), (1 - a)])
#     return y
#
#
# def PCKV_UE(all_kv, d, l, a, p, b):
#     """
#     PCKV论文中的Algorithm2 PCKV-UE
#     :param all_kv:元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
#     :param d:k值维度
#     :param l:虚拟项集长度
#     :param a: a in [0.5,1)
#     :param p: p in [0.5,1)
#     :param b: b in (0,0.5]
#     :return:元素为UE的嵌套表
#     """
#     Y = [P_UE(data, d, l, a, p, b) for data in all_kv]
#     return Y
#
# def AEC_UE(kv_p, d, l, a, p, b):
#     """
#     :param kv_p: UE的输入kv_p是一个嵌套表，表中的元素是二元编码list
#     :param d:
#     :param l:
#     :param a:
#     :param p:
#     :param b:
#     :return:
#     """
#     pos = 0
#     neg = 0
#     n = len(kv_p)
#     n1 = []
#     n2 = []
#     for k in range(d):
#         for kv in kv_p:
#             if kv[k] == 1:
#                 pos += 1
#             elif kv[k] == -1:
#                 neg += 1
#         n1.append(pos)
#         n2.append(neg)
#     f_k = list(l * (-b + (np.array(n1) + np.array(n2)) / n) / (a - b))
#     if f_k < (1 / n):
#         f_k = 1 / n
#     elif f_k > 1:
#         f_k = 1
#
#     A = np.zeros((2, 2))
#     A[0][0] = A[1][1] = a * p - b / 2
#     A[0][1] = A[1][0] = a * (1 - p) - b / 2
#     B = np.zeros((2, 1))
#     B[0, 0] = n1 - n * b / 2
#     B[1, 0] = n2 - n * b / 2
#     S = (A.I) * B
#     n1_ = S[0, 0]
#     n2_ = S[1, 0]
#
#     if n1_ < 0:
#         n1_ = 0
#     elif n1_ > n * f_k / l:
#         n1_ = n * f_k / l
#
#     if n2_ < 0:
#         n2_ = 0
#     elif n2_ > n * f_k / l:
#         n2_ = n * f_k / l
#
#     m_k = l * (n1_ - n2_) / (n * f_k)
#
#     return f_k, m_k
#
#
# """-------------------------------------------"""
#
#
# def P_GRR(k_v, d, l, a, p):
#     """
#     :param k_v: k_v是元组,(k,v),如(3,0.4)
#
#     要重写，这里都写错了
#     :param d:
#     :param l:
#     :param a:
#     :param p:
#     :return:
#     """
#     k = k_v[0]
#     v = k_v[1]
#
#     #b = (1 - p) / ((d + l) - 1) 这个b应该是写错了
#     b=(1-a)/(d+l-1)
#     rnd = np.random.random_sample()
#     if rnd > a - b:
#         k_p = np.random.randint(0, (d + l))
#         v_p = mpp([v, -v], [p, (1 - p)])
#     else:
#         k_p = k
#         v_p = mpp([1, -1], [0.5, 0.5])
#     return k_p, v_p
#
#
# def PCKV_GRR(all_kv, d, l, a, p):
#     y = [P_GRR(data, d, l, a, p) for data in all_kv]
#     return y
#
#
#
#
# def AEC_GRR(kv_p, d, l, a, p, b):
#     """
#
#     :param kv_p: GRR的输入是嵌套表，形如[(1,1),(2,-1),(3,0),(4,1)]
#     :param d:
#     :param l:
#     :param a:
#     :param p:
#     :param b:
#     :return:
#     """
#     pos = 0
#     neg = 0
#     n = len(kv_p)
#     n1 = []
#     n2 = []
#     for k in range(d):
#         for kv in kv_p:
#             if (kv[0] == k) & (kv[1] == 1):
#                 pos += 1
#             elif (kv[0] == k) & (kv[1] == -1):
#                 neg += 1
#         n1.append(pos)
#         n2.append(neg)
#     f_k = list(l * (-b + (np.array(n1) + np.array(n2)) / n) / (a - b))
#     for x in f_k:
#         if x < (1 / n):
#             x = 1 / n
#         elif x > 1:
#             x = 1
#
#     A = np.zeros((2, 2))
#     A[0][0] = A[1][1] = a * p - b / 2
#     A[0][1] = A[1][0] = a * (1 - p) - b / 2
#     B = np.zeros((2, 1))
#     B[0, 0] = n1 - n * b / 2
#     B[1, 0] = n2 - n * b / 2
#     S = (A.I) * B
#     n1_ = S[0, 0]
#     n2_ = S[1, 0]
#
#     if n1_ < 0:
#         n1_ = 0
#     elif n1_ > n * f_k / l:
#         n1_ = n * f_k / l
#
#     if n2_ < 0:
#         n2_ = 0
#     elif n2_ > n * f_k / l:
#         n2_ = n * f_k / l
#
#     m_k = l * (n1_ - n2_) / (n * f_k)
#
#     return f_k, m_k


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
    k = k_v[0]
    v = k_v[1]
    y = [mpp([1, -1, 0], [b / 2, b / 2, (1 - b)]) for i in range(d + l)]

    indx = label.index(k)
    y[indx] = mpp([v, -v, 0], [a * p, a * (1 - p), (1 - a)])
    return y

def P_UE_process(all_kv, d, l, label, a, p, b):
    Y=[P_UE(x, d, l, label, a, p, b) for x in all_kv]
    return Y


def PCKV_UE(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法的改进
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d: 这里的维度与PCKV论文中的不同，这里是candidate的维度，就是2k
    @param l: 填充项长度L
    @param label: 就是candidate+padlable(长度是d+l=2k+L)
    @param a: 真实k保持不变的概率
    @param p: 真实v保持不变的概率
    @param b: 虚假k反转为1的概率
    @return: UE嵌套表list
    """
    Y = [P_UE(x, d, l, label, a, p, b) for x in all_kv]
    return Y

def PCKV_UE_mutilprocess(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法的改进，使用多进程来跑
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d: 这里的维度与PCKV论文中的不同，这里是candidate的维度，就是2k
    @param l: 填充项长度L
    @param label: 就是candidate+padlable(长度是d+l=2k+L)
    @param a: 真实k保持不变的概率
    @param p: 真实v保持不变的概率
    @param b: 虚假k反转为1的概率
    @return: UE嵌套表list
    """
    num_process=8
    kvs=np.array_split(all_kv,num_process)
    for i in range(num_process):
        kvs[i]=kvs[i].tolist()

    pool=Pool()
    res=[pool.apply_async(P_UE_process,args=(kvs[j],d,l,label,a,p,b)) for j in range(num_process)]
    pool.close()
    pool.join()
    Y=[]
    for r in res:
        l=r.get()
        for i in l:
            Y.append(i)
    return Y

def AEC_UE(all_kv_p, d, l, n, a, p, b):
    """
    重写了AEC_UE
    @param all_kv_p: 输入的嵌套list，内容为UE
    @param d: 这里是candidate的维度，就是2k
    @param l: 填充长度L
    @param n: 用户数量
    @param a:
    @param p:
    @param b:
    @return:频率list和均值list
    """
    # n1每一位1的个数，n2每一位是-1的个数
    n1 = []
    n2 = []
    mat = np.array(all_kv_p)

    # line2 计数
    # 这里是d了，不再是d+l了
    for k in range(d):
        tmp = mat[:, k]
        n1.append(sum(tmp == 1))
        n2.append(sum(tmp == -1))
    # print(n1)
    # print(n2)
    n1 = np.array(n1)
    n2 = np.array(n2)

    # line3 计算频率并校正
    f_k_arr = ((n1 + n2) / n - b) * l / (a - b)
    f_k = [correct(1 / n, 1, i) for i in f_k_arr]

    # line4 校正n1,n2
    A = np.zeros((2, 2))
    A[0][0] = A[1][1] = a * p - b / 2
    A[0][1] = A[1][0] = a * (1 - p) - b / 2
    AI = np.matrix(A).I
    B1 = n1 - n * b / 2
    B2 = n2 - n * b / 2
    B = np.vstack((B1, B2))
    S = np.array(AI * B)
    n1_ = S[0]
    n2_ = S[1]
    for k in range(d):
        high = n * f_k[k] / l
        n1_[k] = correct(0, high, n1_[k])
        n2_[k] = correct(0, high, n2_[k])

    # line5 计算均值
    m_k_arr = l * (n1_ - n2_) / (n * np.array(f_k))
    m_k = list(m_k_arr)

    return f_k, m_k

def P_GRR(k_v, d, l, label:list, a, p, b):
    """
    PCKV中相应算法改进
    @param k_v:k_v对
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate+padlable(长度是d+l=2k+L)
    @param a:
    @param p:
    @return:
    """
    k = k_v[0]
    v = k_v[1]
    b = (1 - a) / (d + l - 1)
    randlit=label[:]
    randlit.remove(k)
    rnd = np.random.random_sample()
    if rnd > a:
        k_p = random.choice(randlit)
        v_p = mpp([1, -1], [0.5, 0.5])
    else:
        k_p = k
        v_p = mpp([v, -v], [p, (1 - p)])

    return (k_p, v_p)

def P_GRR_process(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法改进
    @param k_v:k_v对
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate+padlable(长度是d+l=2k+L)
    @param a:
    @param p:
    @return:
    """
    Y=[P_GRR(x, d, l, label, a, p, b) for x in all_kv]
    return Y


def PCKV_GRR(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate+padlable(长度是d+l=2k+L)
    @param a:
    @param p:
    @return:
    """
    Y = [P_GRR(x, d, l, label, a, p, b) for x in all_kv]
    return Y


def PCKV_GRR_mutilprocess(all_kv, d, l, label, a, p, b):
    """
    PCKV中相应算法
    @param all_kv: 元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    @param d:candidate的维度，就是2k
    @param l:填充长度L
    @param label:就是candidate+padlable(长度是d+l=2k+L)
    @param a:
    @param p:
    @return:
    """
    num_process=8
    kvs=np.array_split(all_kv,num_process)
    for i in range(num_process):
        kvs[i]=kvs[i].tolist()
    pool=Pool()
    res=[pool.apply_async(P_GRR_process,args=(kvs[j],d,l,label,a,p,b)) for j in range(num_process)]
    pool.close()
    pool.join()
    Y=[]
    for r in res:
        l=r.get()
        for i in l:
            Y.append(i)
    return Y

def AEC_GRR(all_kv_p, d, l,  n, a, p, b):
    """
    重写了AEC_GRR
    @param label: 就是candidate
    @param all_kv_p: 输入的嵌套list，内容为kv对形如[(1,1),(2,-1),(3,0),(4,1)]
    @param d: 这里是candidate的维度，就是2k
    @param l: 填充长度L
    @param n: 用户数量
    @param a:
    @param p:
    @param b:
    @return:频率list和均值list
    """
    # n1每一位1的个数，n2每一位是-1的个数
    n1 = []
    n2 = []
    for k in range(d):
        # line2 计数
        # 这里是d了，不再是d+l了
        pos = 0
        neg = 0
        for kv in all_kv_p:
            if(kv[0]==k+1):
                if(kv[1]==1):
                    pos+=1
                elif(kv[1]==-1):
                    neg+=1
        n1.append(pos)
        n2.append(neg)
    n1 = np.array(n1)
    n2 = np.array(n2)

    # line3 计算频率并校正
    f_k_arr = ((n1 + n2) / n - b) * l / (a - b)
    f_k = [correct(1 / n, 1, i) for i in f_k_arr]

    # line4 校正n1,n2
    A = np.zeros((2, 2))
    A[0][0] = A[1][1] = a * p - b / 2
    A[0][1] = A[1][0] = a * (1 - p) - b / 2
    AI = np.matrix(A).I
    B1 = n1 - n * b / 2
    B2 = n2 - n * b / 2
    B = np.vstack((B1, B2))
    S = np.array(AI * B)
    n1_ = S[0]
    n2_ = S[1]
    for k in range(d):
        high = n * f_k[k] / l
        n1_[k] = correct(0, high, n1_[k])
        n2_[k] = correct(0, high, n2_[k])

    # line5 计算均值
    m_k_arr = l * (n1_ - n2_) / (n * np.array(f_k))
    m_k = list(m_k_arr)

    return f_k, m_k

def MSE(est, true, d):
    """
    评估函数MSE
    @param est: 估计值list
    @param true: 真实值list
    @param d: list长度
    @return: MSE
    """
    tmp = 0
    for i in range(d):
        tmp += (est[i] - true[i]) ** 2
    mse = tmp / d
    return mse


if __name__ == '__main__':
    # np.random.seed(10)
    # k=[1,2,3,4,5,6,7,8,9]
    # v=[0,.1,.3,.5,.7,.9,-0.2,-0.4,-0.8]
    # kv=list(zip(k,v))
    # d=10
    # l=2
    # kk,vv=PS(kv,d,l)
    # print(kk,vv)

    # d=10
    # l=2
    # b=0.4
    # y = [mpp([1, -1, 0], [b / 2, b / 2, (1 - b)]) for i in range(d + l)]
    # print(y)

    d = 10
    l = 2
    a = 0.6
    p = 0.6
    b = 0.4
    k_v = [(1, -0.2), (2, .5), (3, .3), (4, .1), (5, -0.4), (6, -0.8), (7, 0.3), (8, -0.6), (9, 0.1),(10,1)]

    Y = PCKV_UE(k_v, d, l, a, p, b)
    print(Y)
    # res=[AEC_UE(i,d,l,a,p,b) for i in Y]
    # print(res)

    y = PCKV_GRR(k_v, d, l, a, p)
    print(y)
    res = AEC_GRR(y, d, l, a, p, b)
    print(res)

# 都缺少PS那一步，所以要在主函数中填上PS那一步
