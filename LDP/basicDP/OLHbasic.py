"""
@author:FZX
@file:OLHbasic.py
@time:2020/1/14 7:05 下午
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np
import random
import LDP.basicFunction.basicfunc as bf
import LDP.basicDP.PCKVbasic as pckv
import os

# 这里需要导入xxhash这个包
import xxhash


def p_values(epsilon, n=2):
    """
    计算概率p的值
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    """
    p = np.e ** epsilon / (np.e ** epsilon + n - 1)
    return p


def olh_hash(value, size, seed):
    """
    此olh_hash函数是作者源码里的实现方式
    是使用了xxhash这个包来实现的
    要区别于basicfunc包中我自己实现的hash函数，其实我感觉也差不多
    :param value:输入的值
    :param size: 哈希域的大小,就是g
    :param seed: 种子,原文作者将用户的编号作为种子，即第一个用户为0，下一个为1
    :return: 哈希得到的值，在[0，g）之间
    """

    h = (xxhash.xxh32(str(value), seed=seed).intdigest()) % size
    return h


# def grr(p, bit, g):
#     v = [i for i in range(g)]
#     rnd = np.random.random()
#     if rnd <= p:
#         perturbed_bit = bit
#     else:
#         del (v[bit])
#         perturbed_bit = random.choice(v)
#     return perturbed_bit


def olh_grr(p, bit, g):
    """
    olh_grr是作者源码里实现扰动的方式
    :param p: 保持不变的概率，即exp(epsilon) / (exp(epsilon) + g - 1)
    :param bit: 原值
    :param g: 哈希空间
    :return: 扰动后的值
    """
    q = (1 - p) / (g - 1)
    rnd = np.random.random_sample()
    if rnd > p - q:
        perturbed_bit = np.random.randint(0, g)
    else:
        perturbed_bit = bit
    return perturbed_bit


# def support(v, report, g, n):
#     """
#     这个support函数是和我自己写的bf下的hash相匹配
#     :param v:
#     :param report:
#     :param g:
#     :param n:
#     :return:
#     """
#     c = 0
#     for j in range(n):
#         if bf.hash(v, g, j) == report[j]:
#             c = c + 1
#     return c


def olh_support(v, y, g, n):
    """
    这个support函数是和作者的olh_hash相匹配
    @param v: 要查询的项
    @param y: 报告向量y
    @param g:
    @param n: 用户数量
    @return:
    """
    c = 0
    for i in range(n):
        if olh_hash(v, g, i) == y[i]:
            c += 1
    return c



def aggregation(c, n, g, p):
    """
    估计频率函数
    :param c: support 输入为x的报告的个数
    :param n: 用户数
    :param g: hash后的维度
    :param p: 概率p
    :return:
    """
    est = (c - n / g) / (p - 1 / g)
    return est


def aggregator(item, y: list, g: int, n: int, p: int):
    """
    将olh_support和aggregation封装在一起
    @param item: 需要求的item的编号 即item in label
    @param y: 扰动后的结果y
    @param g: hash域的大小
    @param n: 用户数
    @param p: 反转概率
    @return: 估计值list
    """
    c = olh_support(item, y, g, n)
    estimate = (c - n / g) / (p - 1 / g)
    # estimate = float('{:.3f}'.format(estimate))
    estimate = float('%.3f' % estimate)  # 结果保留3位小数
    return estimate


# 封装成OLH
def OLH_1(epsilon: int, valuelist: list, n: int, l: int):
    """
    标签标准化成[1,l]之后就可以用这个函数
    @param l: l是key的维度
    @param n: 用户数量
    @param valuelist: 用户上传的list，从每个用户那里采样1个项
    @param epsilon: 隐私预算epsilon
    @return:频率估计结果
    """

    g = int(np.exp(epsilon)) + 1  # 参数g
    p = p_values(epsilon, n=g)  # 参数p

    """
    Perturbing
    哈希：x存放的是valuelist的值hash之后的结果，在区间[0,g)中
    扰动：y存放的是x的值扰动之后的结果，在区间[0,g)中
    """
    # x = [olh_hash(valuelist[i], g, i) for i in range(n)]  # 这种写法速度快
    # y = [olh_grr(p, x[i], g) for i in range(n)]

    y = [olh_grr(p, olh_hash(valuelist[i], g, i), g) for i in range(n)]

    """
    Aggregation

    """
    estimat = [(i + 1, aggregator(i + 1, y, g, n, p)) for i in range(l)]

    return estimat


# 封装成OLH
def OLH(epsilon, valuelist: list, n: int, label: list):
    """
    封装后的OLH,label没有规范化的用这个函数
    @param label: 标签list，其实就所有在valuelist中出现过的项组成的list
    @param n: 用户数量
    @param valuelist: 用户上传的list，从每个用户那里采样1个项
    @param epsilon: 隐私预算epsilon
    @return:频率估计结果list
    """

    g = int(np.exp(epsilon)) + 1  # 参数g
    p = p_values(epsilon, n=g)  # 参数p

    """
    Perturbing
    哈希：x存放的是valuelist的值hash之后的结果，在区间[0,g)中
    扰动：y存放的是x的值扰动之后的结果，在区间[0,g)中
    """
    x = [olh_hash(valuelist[i], g, i) for i in range(n)]  # 这种写法速度快
    y = [olh_grr(p, x[i], g) for i in range(n)]

    """
    Aggregation
    
    """
    estimat = [(label[i], aggregator(label[i], y, g, n, p)) for i in range(len(label))]

    return estimat


# OLH扰动函数
def OLH_perturbation(epsilon: int, value: int, seed: int):
    """
    OLH扰动的函数
    @param epsilon: 隐私预算
    @param value: 值
    @param seed: 种子，用户编号作为种子，第一个用户0，下一个为1
    @return: 扰动后的值，在[0,g)之间
    """
    g = int(np.exp(epsilon)) + 1  # 参数g
    p = p_values(epsilon, n=g)  # 参数p
    h = olh_hash(value, g, seed)
    x = olh_grr(p, h, g)
    return x


def OLH_aggregation_multithread(epsilon: int, perturbed_value: list, n:int,d: int, num_thread,func):
    """
    使用多线程来计算聚合
    @param func:跑在线程里的函数
    @param epsilon: 隐私预算
    @param perturbed_value: 上传的扰动值list
    @param n: 用户数
    @param d: 数据域的维度
    @param num_thread: 线程数
    @return: 估计值list
    """
    g = int(np.exp(epsilon)) + 1  # 参数g
    p = p_values(epsilon, n=g)  # 参数p
    pvs = np.array_split(perturbed_value, num_thread) # 切分成相应的线程数，即每个线程就是一个边缘节点
    for i in range(num_thread):
        pvs[i] = pvs[i].tolist()

    with ThreadPoolExecutor(max_workers=11) as pool:
        results = pool.map(func, pvs, [g for i in range(num_thread)],
                           [d for i in range(num_thread)])
    temp = []
    for r in results:
        print(r)
        temp.append(r)
    support = np.sum(temp, axis=0)

    est = aggregation(support, n,g, p).tolist()
    return est

def OLH_aggregation_mutilprocess(epsilon: int, perturbed_value: list, n:int,d: int ,func,num_process=6):
    """
    多进程，火力全开
    @param epsilon: 隐私预算
    @param perturbed_value: 上传的扰动值list
    @param n: 用户数
    @param d: 数据域的维度
    @param func: 跑在进程里的函数
    @param num_process: 分片数，默认是6个，因为我的CPU是8核，所以开6个核心同时跑
    @return:
    """
    g = int(np.exp(epsilon)) + 1  # 参数g
    p = p_values(epsilon, n=g)  # 参数p
    pvs = np.array_split(perturbed_value, num_process) # 切分成相应的线程数，即每个线程就是一个边缘节点
    for i in range(num_process):
        pvs[i] = pvs[i].tolist()
    pool=Pool()

    res=[pool.apply_async(func,args=([pvs[j],g,d,])) for j in range(num_process)]
    pool.close()
    pool.join()
    temp=[]
    for r in res:
        #print(r.get())
        temp.append(r.get())
    support=np.sum(temp,axis=0)
    est = aggregation(support, n,g, p).tolist()
    return est


def get_support(y: list, g, d):
    """
    计算support值
    @param y:
    @param g:
    @param n:
    @param d:
    @return:
    """
    print(multiprocessing.current_process().name,"启动")
    support = []
    for i in range(d):
        c = 0
        for j in range(len(y)):
            if olh_hash(i+1, g, y[j][1]) == y[j][0]:
                c += 1
        support.append(c)
    return support

