# -*- coding:utf-8 -*-
"""
@author:FZX
@file:epslion.py
@time:2019/1/23 15:41
"""
#泊松分布分配epsilon
import math
x=[]
eps=1
k=10
def fun(i,k):
    return(eps*(k**(i-1)*(math.exp(-k)/math.factorial(i-1))+eps*(k**(i+k-1)*(math.exp(-k)/math.factorial(i+k-1)))))

for i in range(1,11):
   print(fun(i,k))



