"""
@author:FZX
@file:SHbasic.py
@time:2019/11/25 20:46
"""
import numpy as np

def gen_m_bit_string(bits):
    m=len(bits)
    down=1/np.sqrt(m)
    print(down)
    bits[bits==0]=-1*down
    bits[bits==1]=down
    return bits

x=np.array([0.,1.,0.,1.,0.,0.,1.,0.,1.,1.,0.],dtype=np.float16)
s=gen_m_bit_string(x)
print(s)

