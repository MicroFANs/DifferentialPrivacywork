# -*- coding:utf-8 -*-
"""
@author:FZX
@file:creatdataset.py
@time:2019/3/22 19:01
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd


def GeneratePointInCycle4(point_num, radius,xx,yy):
    x1=[]
    y1=[]
    for i in range(1, point_num + 1):
        theta = random.random() * 2 * pi;
        r = random.uniform(0, radius)
        x = xx+math.sin(theta) * (r ** 0.5)
        y = yy+math.cos(theta) * (r ** 0.5)
        x1.append(x)
        y1.append(y)
        #plt.plot(x, y, '*', color="black")
    return x1,y1


pi = np.pi
theta = np.linspace(0, pi * 2, 1000)
R = 1
x = np.sin(theta) * R
y = np.cos(theta) * R

#plt.figure(figsize=(6, 6))
#plt.plot(x, y, label="cycle", color="red", linewidth=2)
plt.title("cycyle")
x1,y1=GeneratePointInCycle4(1500, 2,0,0)  # 修改此处来显示不同算法的效果
x2,y2=GeneratePointInCycle4(1500,2,5,0)
x3,y3=GeneratePointInCycle4(1500,2,0,5)
x4,y4=GeneratePointInCycle4(1500,2,5,5)
# plt.plot(x1,y1,'o')
# plt.plot(x2,y2,'*')
x1.extend(x2)
x3.extend(x4)
x1.extend(x3)
y1.extend(y2)
y3.extend(y4)
y1.extend(y3)
# plt.plot(x1,y1,'o')
# plt.legend()
# plt.show()

z=np.zeros((6000,2))
for i in  range(6000):
    z[i,:]=[x1[i],y1[i]]
print(z)
savez=pd.DataFrame(z)
#savez.to_csv('D:\Git\DifferentialPrivacywork\dataset/at1.csv',header=None,index=False)
np.random.shuffle(z)
print(z)

# savex=pd.DataFrame(x1)
# savey=pd.DataFrame(y1)
savez1=pd.DataFrame(z)
# savex.to_csv('D:\Git\DifferentialPrivacywork\experiment2\output/x.csv')
# savey.to_csv('D:\Git\DifferentialPrivacywork\experiment2\output/y.csv')
savez1.to_csv('D:\Git\DifferentialPrivacywork\dataset/at1.csv',header=None,index=False)
