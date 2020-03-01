"""
@author:FZX
@file:PCKVbasic.py
@time:2020/2/29 18:33
"""
import numpy as np


def p_values(epsilon, n=2):
    """
    计算概率p的值
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    """
    p = np.e ** epsilon / (np.e ** epsilon + n - 1)
    return p

def PS(k,v,d,l):
    """
    PCKV论文中的Algorithm1 Padding and Sampling
    以高概率b从用户的项集S中采样，挑选一个kv对
    以低概率（1-b）从扩充的虚拟项集l中挑选虚拟k值，并且v值值0
    此时总的k值的维度也从d变成了(d+l)
    :param k:用户的k值list
    :param v:用户的v值list
    :param d:整个k值的值域的维度
    :param l:虚拟项的长度
    :return:
    """
    S=len(k)
    b=S/(max(S,l))
    rnd=np.random.random()
    k_ps=0 #ps之后的k值
    v_ps=0 #ps之后的v值
    if rnd<b: # 从用户项集S中随机选择一个kv对，从[0,S)中随机选择一个数作为序号
        j=np.random.randint(0,S) # 前闭后开区间
        k_ps=j+1 # 序号+1才是真正的k值
        v_ps=v[j]
    else:
        v_ps=0
        j=np.random.randint(0,l)+d # 从{d+1,d+2,...,d'}中随机选择一个作为序号
        k_ps=j+1 # 序号+1才是真正的k值

    # 离散v_ps值
    p=(1+v_ps)/2
    rnd=np.random.random()
    if rnd<p:
        v_ps=1
    else:
        v_ps=-1

    return k_ps,v_ps



def mpp(candidate, p):
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

def P_UE(k_v,d,l,a,p,b):
    """
    PCKV_UE算法中的扰动部分
    :param k_v: k_v是元组,(k,v),如(3,0.4)
    :param d:
    :param l:
    :param a:
    :param p:
    :param b:
    :return:
    """
    y=[mpp([1,-1,0],[b/2,b/2,(1-b)]) for i in range(d+l)]
    k=k_v[0]
    v=k_v[1]
    y[k-1]=mpp([v,-v,0],[a*p,a*(1-p),(1-a)])
    return y


def PCKV_UE(all_kv,d,l,a,p,b):
    """
    PCKV论文中的Algorithm2 PCKV-UE
    :param all_kv:元素为元组的list 如[(2,.5),(3,.3),(4,.1),(5,-0.4)]
    :param d:k值维度
    :param l:虚拟项集长度
    :param a: a in [0.5,1)
    :param p: p in [0.5,1)
    :param b: b in (0,0.5]
    :return:元素为UE的嵌套表
    """
    Y=[P_UE(data,d,l,a,p,b) for data in all_kv]
    return Y


def P_GRR(k_v,d,l,a,p):
    """

    :param k_v: k_v是元组,(k,v),如(3,0.4)
    :param d:
    :param l:
    :param a:
    :param p:
    :return:
    """
    k = k_v[0]
    v = k_v[1]

    b = (1 - p) / ((d+l) - 1)
    rnd = np.random.random_sample()
    if rnd > a - b:
        k_p = np.random.randint(0, (d+l))
        v_p=mpp([v,-v],[p,(1-p)])
    else:
        k_p = k
        v_p=mpp([1,-1],(0.5,0.5))
    return k_p,v_p



def PCKV_GRR(all_kv,d,l,a,p):
    y=[P_GRR(data,d,l,a,p) for data in all_kv]
    return y




if __name__ == '__main__':
    # np.random.seed(10)
    # k=[1,2,3,4,5,6,7,8,9]
    # v=[0,.1,.3,.5,.7,.9,-0.2,-0.4,-0.8]
    # d=12
    # l=2
    # kk,vv=PS(k,v,d,l)
    # print(kk,vv)



    # d=10
    # l=2
    # b=0.4
    # y = [mpp([1, -1, 0], [b / 2, b / 2, (1 - b)]) for i in range(d + l)]
    # print(y)

    d=10
    l=2
    a=0.6
    p=0.6
    b=0.4
    k_v=[(1,-0.2),(2,.5),(3,.3),(4,.1),(5,-0.4),(6,-0.8),(7,0.3),(8,-0.6),(9,0.1)]

    Y=PCKV_UE(k_v,d,l,a,p,b)
    print(Y)

    y=PCKV_GRR(k_v,d,l,a,p)
    print(y)


# 都缺少PS那一步，所以要在主函数中填上PS那一步