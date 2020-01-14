"""
@author:FZX
@file:OLHbasic.py
@time:2020/1/14 7:05 下午
"""

import numpy as np
import random


def grr(p,bit,g):
    v= [i for i in range(g)]
    rnd=np.random.random()
    if rnd<=p:
        perturbed_bit=bit
    else:
        del(v[bit])
        perturbed_bit=random.choice(v)
    return perturbed_bit

