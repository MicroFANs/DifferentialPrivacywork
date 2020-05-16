"""
@author:FZX
@file:tool.py
@time:2020/5/16 4:10 下午
"""
import LDP.basicFunction.basicfunc as bf
import re



path='./data/Clothing_k.txt'
lpath='./data/Clothing_lable.txt'
data=bf.readtxt(path,'int')
print(len(data))