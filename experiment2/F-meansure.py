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
kmeansdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output\BloodDataSetresult_normal.csv',header=None)
DPdata=pd.read_csv('D:\Git\DifferentialPrivacywork\experiment2\output/2019-03-14-16_15_23DPBlood_out.csv',header=None)

#origin=np.array(origindata[4])
kmeans=np.array(kmeansdata[4])
DP=np.array(DPdata[4])

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

    # recall precision f-measure
    # print ('precision:%.3f' %precision_score(y_true=kmeans, y_pred=DP1))
    # print ('recall:%.3f' %recall_score(y_true=kmeans, y_pred=DP1))
    # print ('F1:%.3f' %f1_score(y_true=kmeans, y_pred=DP1))
    # print(precision_recall_fscore_support(y_true=kmeans, y_pred=DP1, average='macro'))

    measurelist=classification_report(y_true=y_true, y_pred=y_pred)
    print(measurelist)
    return 0

# measure(origin,kmeans)
measure(kmeans,DP)