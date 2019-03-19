# -*- coding:utf-8 -*-
"""
@author:FZX
@file:F-meansure.py
@time:2019/3/12 14:25
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


# 数据
#origindata=pd.read_csv('D:\Git\DifferentialPrivacywork\dataset\Iris_normal_lable.csv',header=None)
kmeansdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8result_normal.csv',header=None)
avgDPdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8avgDP2_7iters.csv',header=None)
div2DPdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8div2DP2_6iters.csv',header=None)
myDPdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\h8\h8myDP2_8iters.csv',header=None)
index=kmeansdata.shape[1]
kmeans=np.array(kmeansdata[index-1])
avgDP=np.array(avgDPdata[index-1])
div2DP=np.array(div2DPdata[index-1])
myDP=np.array(myDPdata[index-1])

def measure(y_true,y_pred):
    # 混淆矩阵
    confmat=confusion_matrix(y_true,y_pred)
    print(confmat)
    #plt.matshow(confmat, cmap=plt.cm.gray)
    #plt.show()

    fig,ax = plt.subplots(figsize=(7,7))
    cax=ax.matshow(confmat,cmap=plt.cm.Blues,alpha=1)
    #fig.colorbar(cax)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    plt.show()
    measurelist=classification_report(y_true=y_true, y_pred=y_pred)
    print(measurelist)
    return 0

measure(kmeans,avgDP)
measure(kmeans,div2DP)
measure(kmeans,myDP)