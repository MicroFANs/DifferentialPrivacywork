"""
@author:FZX
@file:basicfunc.py
@time:2019/12/3 20:10
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def readcsv(path):
    """
    读取csv文件，并转换成numpy数组输出
    :param path: 路径
    :return: numpy数组格式
    """
    df_data=pd.read_csv(path,header=-1)
    np_data=np.array(df_data)
    return np_data


def savecsv(nparry,path,header=False,index=False):
    """
    将numpy数组保存成csv文件
    :param nparry: numpy数组
    :param path: 保存路径
    :param header: 行名，默认为否
    :param index: 列编号，默认为否
    :return:
    """
    df=pd.DataFrame(nparry)
    df.to_csv(path,header=header,index=index)



def one_hot_1D(np_data):
    """
    一维数据的onehot编码，首先label是数据域中所有数据的list，list_data就是所要编码的一位数据的list
    :param np_data: numpy数组数据
    :return: 一个numpy数据onehot矩阵，行数是用户数，列数是数据的数，用户有那个数，那一位就置为1
    """
    max = np.max(np_data) # 一维数据中的最大值
    min = np.min(np_data) # 一维数据中的最小值
    n=len(np_data) # 用户i的数量n
    d=max-min+1 # 用户数据域的维度d

    label=np.arange(1,max+1).reshape(max,1).tolist()
    onehot=OneHotEncoder()
    onehot.fit(label)
    list_data=np_data.tolist()
    onehot_data=onehot.transform(list_data).toarray()
    return onehot_data


def Relative_Error(actual,estimate):
    """
    LDPMiner论文中的相对误差指标，用来评价top-k中估计频率的误差
    :param actual: heavyhitter的真实频率，numpy数组
    :param estimate: 频率估计，numpy数组
    :return: 相对误差RE
    """
    error=np.abs(estimate-actual)/actual
    RE=np.median(error)
    return RE


def rel(d,k,rank_true,rank_estimate):
    """
    LDPMiner论文中排名误差指标DCG中的参数rel
    :param d: 数据维度
    :param k: topk的k
    :param rank_true:真实排序 长度>=k
    :param rank_estimate: 估计得到的排序 长度>=k
    :return: rel数组，真实排序中每个项的rel值
    """
    topk_act=np.zeros(k) # 真实topk中每个元素v的rank值
    topk_est=np.zeros(k) # 估计得到的topk中对应的元素v的rank值
    for i in range(k):
        v=rank_true[i] # 遍历真实排序rank_true中的前k个项

        index_true=np.where(rank_true==v)
        topk_act[i]=index_true[0][0]+1   # 生成真实topk项的rank值

        index_est=np.where(rank_estimate==v) # 生成估计topk项的rank值
        topk_est[i]=index_est[0][0]+1

    print(topk_act,topk_est)
    rel=np.log2(np.abs(d-np.abs(topk_act-topk_est)))
    return rel

def DCG(d,k,rank_true,rank_estimate):
    rel_v=rel(d,k,rank_true,rank_estimate)
    print(rel_v)
    sum=rel_v[0]
    for i in range(1,k):
        temp=rel_v[i]/np.log2(i+1)
        sum=sum+temp
    dcg=sum

    return dcg

def IDCG(d,k):
    rel_v=np.log2(d)*np.ones(k)
    print(rel_v)
    sum=rel_v[0]
    for i in range(1,k):
        temp=np.log2(d)/np.log2(i+1)
        sum=sum+temp
    idcg=sum

    return idcg



true=np.array([1,2,3,4,5,6,7,8,9,10])

est=np.array([4,2,1,3,5,7,10,8,6,9])




dcg=DCG(64,6,true,est)
idcg=IDCG(64,6)

print('DCG',dcg)
print('IDCG',idcg)

