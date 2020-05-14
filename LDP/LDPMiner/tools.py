"""
@author:FZX
@file:tools.py
@time:2020/1/3 4:51 下午
"""

import LDP.basicFunction.basicfunc as bf
import linecache


def gen_l_set(l):
    """
    生成填充之后的l_set,都用0补全为长度l的项集
    :param l:l_set的长度
    :return:no return
    """
    # 读取数据
    path='../LDPMiner/dataset/kosarak/kosarak.txt'
    user_data=bf.readtxt(path)
    #sampling过程，构造私有项集
    savepath='../LDPMiner/dataset/kosarak/l_set.txt'
    v_list=bf.gen_iterm_set(user_data,l)
    #print(type(v_list))
    bf.savetxt(v_list,savepath)




if __name__ == '__main__':
    gen_l_set(l=20)
