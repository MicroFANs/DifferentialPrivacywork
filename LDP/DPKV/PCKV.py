"""
@author:FZX
@file:PCKV.py
@time:2020/2/19 18:57
"""
import matplotlib

matplotlib.use('TkAgg')
import LDP.basicDP.RPbasic as rpb
import LDP.basicDP.SHbasic as shb
import LDP.basicDP.OLHbasic as lhb
import LDP.basicFunction.basicfunc as bf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

# 关闭科学计数法显示
np.set_printoptions(suppress=True)


x=bf.readtxt('/Workplace\pyworkplace\DifferentialPrivacywork\dataset\KV\KV_v.txt')


