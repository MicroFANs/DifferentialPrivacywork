"""
@author:FZX
@file:mybasic.py
@time:2019/11/25 19:44
"""

# some LDP basic functions

import numpy as np

def p_values(epsilon,n=2):
    '''
    计算概率p的值,在RAPPOR中的敏感度需要从epsilon来体现，如果敏感度为h,则预算应该为(epsilon/h)
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    '''
    p=np.e**epsilon/(np.e**epsilon+n-1)
    return p