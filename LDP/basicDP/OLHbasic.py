"""
@author:FZX
@file:OLHbasic.py
@time:2020/1/14 7:05 下午
"""

import numpy as np
import random
import LDP.basicFunction.basicfunc as bf


def grr(p,bit,g):
    v= [i for i in range(g)]
    rnd=np.random.random()
    if rnd<=p:
        perturbed_bit=bit
    else:
        del(v[bit])
        perturbed_bit=random.choice(v)
    return perturbed_bit


def support(v,report,g,n):
    c=0
    for j in range(n):
        if bf.hash(v,g,j)==report[j]:
            c=c+1
    return c

def aggregation(c,n,g,p):
    """
    估计频率函数
    :param c: support 输入为x的报告的个数
    :param n: 用户数
    :param g: hash后的维度
    :param p: 概率p
    :return:
    """
    est=(c-n/g)/(p-1/g)
    return est
