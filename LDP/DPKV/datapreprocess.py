"""
@author:FZX
@file:datapreprocess.py
@time:2020/2/19 21:58
"""

import LDP.basicFunction.basicfunc as bf
import numpy as np


def normlization(data):
    """
    归一化到[-1,1]
    :param data:
    :return:
    """
    max = np.max(data)
    min = np.min(data)
    lenth = max - min
    norm = -1 + 2 * (data - min) / lenth
    return np.around(norm, 2)


"""
def KV(path):

    # 处理KV数据的笨方法，耗时太长，不要用

    user_data = bf.readcsv(path)
    n = 7120  # 总共有7120个用户
    # 用户的kv值用list来装
    k = []
    v = []

    for id in range(n):
        temp_k = []  # 用temp_k来保存每个id的k值
        temp_v = []  # 用temp_v来保存每个id的v值
        for i in range(len(user_data)):
            if user_data[i][0]==id+1:
                temp_k.append(user_data[i][1])
                temp_v.append(user_data[i][2])
        k.append(temp_k)
        v.append(temp_v)
    print(k)
    print(v)
"""


def KV(path):
    """
    自己造的数据集KV,n=7120
    :param path:
    :return:
    """

    data = bf.readcsv(path)
    v_ary = data[:, 2]
    v_ary = normlization(v_ary)
    # 对value值进行归一化
    data[:, 2] = v_ary

    n = 7120  # 总共有7120个用户

    # 创建嵌套表，每个list里有n个list，即对应每个用户的不定长的数据
    k = [[] for i in range(n)]  # 装key值
    v = [[] for i in range(n)]  # 装value值

    # index用来存放每个用户id的数据的索引，之后用这个索引来填k[]和v[]，
    # 也是个嵌套表，只不过k[]v[]装的是数据，index[]装的是索引
    index = []

    for id in range(n):
        temp = np.argwhere(data[:, 0] == id + 1)
        t = (temp.flatten()).tolist()
        index.append(t)

    for id in range(n):
        l = index[id]  # l是每个第id个用户拥有的数据行索引的list
        for i in range(len(l)):
            tp = l[i]  # tp是行号
            k[id].append(data[tp][1])
            v[id].append(data[tp][2])

    print(index[1])
    print(k[0])
    print(v[0])
    bf.savetxt(k, '/Workplace\pyworkplace\DifferentialPrivacywork\dataset\KV\KV_k.txt')
    bf.savetxt(v, '/Workplace\pyworkplace\DifferentialPrivacywork\dataset\KV\KV_v.txt')


if __name__ == '__main__':
    path = '/Workplace\pyworkplace\DifferentialPrivacywork\dataset\KV\KV.csv'
    KV(path)
