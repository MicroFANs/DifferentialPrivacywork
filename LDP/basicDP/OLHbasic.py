"""
@author:FZX
@file:OLHbasic.py
@time:2020/1/14 7:05 下午
"""

import numpy as np
import random
import LDP.basicFunction.basicfunc as bf

# 这里需要导入xxhash这个包
import xxhash


def olh_hash(value, size, seed):
    """
    此olh_hash函数是作者源码里的实现方式
    是使用了xxhash这个包来实现的
    要区别于basicfunc包中我自己实现的hash函数，其实我感觉也差不多
    :param value:输入的值
    :param size: 哈希域的大小,就是g
    :param seed: 种子,原文作者将用户的编号作为种子，即第一个用户为0，下一个为1
    :return: 哈希得到的值，在（0，g）之间
    """

    h=(xxhash.xxh32(str(value), seed=seed).intdigest() % size)
    return h


def grr(p, bit, g):
    v = [i for i in range(g)]
    rnd = np.random.random()
    if rnd <= p:
        perturbed_bit = bit
    else:
        del (v[bit])
        perturbed_bit = random.choice(v)
    return perturbed_bit


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


def support(v, report, g, n):
    """
    这个support函数是和我自己写的bf下的hash相匹配
    :param v:
    :param report:
    :param g:
    :param n:
    :return:
    """
    c = 0
    for j in range(n):
        if bf.hash(v, g, j) == report[j]:
            c = c + 1
    return c

def olh_support(v, report, g, n):
    """
    这个support函数是和我自己写的bf下的hash相匹配
    :param v:
    :param report:
    :param g:
    :param n:
    :return:
    """
    c = 0
    for j in range(n):
        if olh_hash(v, g, j) == report[j]:
            c = c + 1
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
