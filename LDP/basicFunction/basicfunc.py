"""
@author:FZX
@file:basicfunc.py
@time:2019/12/3 20:10
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random as rnd
import itertools


def create_0_1(row, col):
    """
    生成row行 col列的随机0和1numpy数组
    :param row: 行数
    :param col: 列数
    :return:
    """
    length = row * col
    array = np.random.randint(0, 2, length).reshape(row, col)
    return array


def readcsv(path, N=None):
    """
    读取csv文件，并转换成numpy数组输出
    :param path: 路径
    :param N: 读取前N行，默认是None
    :return: 数组格式
    """
    df_data = pd.read_csv(path, header=-1, nrows=N)
    np_data = np.array(df_data)
    return np_data


def savecsv(nparry, path, header=False, index=False):
    """
    将numpy数组保存成csv文件
    :param nparry: numpy数组
    :param path: 保存路径
    :param header: 行名，默认为否
    :param index: 列编号，默认为否
    :return:
    """
    df = pd.DataFrame(nparry)
    df.to_csv(path, header=header, index=index)


def readtxt(path):
    """
    按行读取数据，放进list中，list的元素又是一个list，表示原txt中每行的数据，例如[[1,2],[2,3,4,5,1],[1,4,3],[1,2,4]]
    :param path:路径
    :return:返回值是list
    """
    f = open(path)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    return data_list

def savetxt(data,path):
    file = open(path,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def combine_lists(lists):
    """
    将元素为list的list进行合并，成为一个列表
    :param lists: 元素为list的list，如[[1,2],[3,4],[5],[1,2,3,4,5]]
    :return: 合并后的列表[1,2,3,4,5,1,2,3,4,5]
    """
    res_list=list(itertools.chain.from_iterable(lists))
    return res_list


def gen_iterm_set(data_list,l):
    """
    生成每个用户的项集，长度为l，对于原项集长度大于l的从中随机选择l个，原项集小于l的，打乱原顺序再补0
    :param data_list: 用户原始数据list
    :param l: 项集v长度
    :return: n*l的总项集，每一行为一个用户的项集，长度为l，格式为list
    """
    v_list = []
    n=len(data_list)
    for i in range(n):
        x = data_list[i]
        length=len(x)
        if length <l:
            zero_list = [0 for index in range(l - length)]
            rnd.shuffle(x)
            x.extend(zero_list)

            v_list.append(x)
        else:
            v_list.append(rnd.sample(x, l))

    return v_list


def one_hot_1D(np_data):
    """
    一维数据的onehot编码，首先label是数据域中所有数据的list，list_data就是所要编码的一位数据的list
    :param np_data: numpy数组数据
    :return: 一个numpy数据onehot矩阵，行数是用户数，列数是数据的数，用户有那个数，那一位就置为1
    """
    max = np.max(np_data)  # 一维数据中的最大值
    min = np.min(np_data)  # 一维数据中的最小值
    n = len(np_data)  # 用户i的数量n
    d = max - min + 1  # 用户数据域的维度d

    label = np.arange(1, max + 1).reshape(max, 1).tolist()
    onehot = OneHotEncoder()
    onehot.fit(label)
    list_data = np_data.tolist()
    onehot_data = onehot.transform(list_data).toarray()
    return onehot_data

def one_hot(np_data,label):
    """
    onehot编码，这个函数的好处是，可以根据输入的label进行编码。上面那个函数只能由1到n编码（即，leable默认为1-n的数组）
    :param np_data: 输入数据，二维数组，格式为n*1或者n*k
    :param label: 编码的标签，就是所有候选值的数组，用来对应相应的位置
    :return: onehot编码，np二维数组
    """
    d=len(label)
    n=len(np_data)
    code=np.zeros((n,d+1))
    for i in range(n):
        if np_data[i][0]==0:
            index=0
        else:
            index_arr=np.argwhere(label==np_data[i][0])
            index=index_arr[0][0]+1
        code[i][index]=1
    onehotcode=code[:,1:d+1]
    return onehotcode



def Relative_Error(actual, estimate):
    """
    LDPMiner论文中的相对误差指标，用来评价top-k中估计频率的误差
    :param actual: heavyhitter的真实频率，numpy数组
    :param estimate: 频率估计，numpy数组
    :return: 相对误差RE
    """
    error = np.abs(estimate - actual) / actual
    RE = np.median(error)
    return RE


def rel(d, k, rank_true, rank_estimate):
    """
    LDPMiner论文中排名误差指标DCG中的参数rel
    :param d: 数据维度
    :param k: topk的k
    :param rank_true:真实排序 长度>=k
    :param rank_estimate: 估计得到的排序 长度>=k
    :return: rel数组，真实排序中每个项的rel值
    """
    topk_act = np.zeros(k)  # 真实topk中每个元素v的rank值
    topk_est = np.zeros(k)  # 估计得到的topk中对应的元素v的rank值
    for i in range(k):
        v = rank_true[i]  # 遍历真实排序rank_true中的前k个项

        index_true = np.where(rank_true == v)
        topk_act[i] = index_true[0][0] + 1  # 生成真实topk项的rank值

        index_est = np.where(rank_estimate == v)  # 生成估计topk项的rank值
        topk_est[i] = index_est[0][0] + 1

    print(topk_act, topk_est)
    rel = np.log2(np.abs(d - np.abs(topk_act - topk_est)))
    return rel


def DCG(d, k, rank_true, rank_estimate):
    rel_v = rel(d, k, rank_true, rank_estimate)
    print(rel_v)
    sum = rel_v[0]
    for i in range(1, k):
        temp = rel_v[i] / np.log2(i + 1)
        sum = sum + temp
    dcg = sum

    return dcg


def IDCG(d, k):
    rel_v = np.log2(d) * np.ones(k)
    print(rel_v)
    sum = rel_v[0]
    for i in range(1, k):
        temp = np.log2(d) / np.log2(i + 1)
        sum = sum + temp
    idcg = sum

    return idcg



# hash函数，是布隆滤波器的实现中的
def hash(data,seed):
    hash=0
    length=len(data)
    if length>0:
        for i in range(length):
            hash=i*hash+data[i]
    hash=hash*seed%16 # size的值
    return abs(hash)



if __name__ == '__main__':
    # true = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10,11])
    #
    # est = np.array([4, 2, 11, 3, 5, 7, 10, 8, 6, 9])
    #
    # dcg = DCG(64, 6, true, est)
    # idcg = IDCG(64, 6)
    #
    # print('DCG', dcg)
    # print('IDCG', idcg)



    # path = '../LDPMiner/dataset/kosarak/kosarak_10k.txt'
    # data_list1=readtxt(path)
    # v_list1=gen_iterm_set(data_list1,12,6)
    # print(len(v_list1))



    # 测试hash
    x=np.array([1,2,3])
    size=16
    seeds=[ 2, 3, 5, 7]
    indexs=[]
    index=0
    for i in range(len(seeds)):
        index=hash(x,seeds[i])
        print(index)


