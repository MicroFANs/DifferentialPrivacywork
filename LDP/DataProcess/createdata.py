"""
@author:FZX
@file:createdata.py
@time:2019/12/4 21:24
"""
import numpy as np
import LDP.basicFunction.basicfunc as bf


def create_0_1(row,col):
    """
    生成row行 col列的随机0和1numpy数组
    :param row: 行数
    :param col: 列数
    :return:
    """
    length=row*col
    array=np.random.randint(0,2,length).reshape(row,col)
    return array




















if __name__ == '__main__':
    array=create_0_1(10000,1)
    print(array)
    path='../LDPMiner/dataset/SH/10k_singlevalue.csv'
    bf.savecsv(array,path=path)
