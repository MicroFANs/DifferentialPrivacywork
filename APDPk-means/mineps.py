# -*- coding:utf-8 -*-
"""
@author:FZX
@file:mineps.py
@time:2019/3/20 15:52
"""
import numpy as np

def mineps(k,N,d):
    minepslion=np.sqrt(((500*k**3)/N**2)*np.power((d+(4*d*0.45**2)**(1/3)),3))
    result=round(minepslion,3)
    return result

print(mineps(k=2,N=4000,d=2))