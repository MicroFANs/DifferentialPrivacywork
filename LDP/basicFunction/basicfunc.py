"""
@author:FZX
@file:basicfunc.py
@time:2019/12/3 20:10
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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