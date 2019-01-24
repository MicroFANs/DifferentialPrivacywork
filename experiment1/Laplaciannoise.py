# Implementing Laplace mechanism on Adult dataset by adding Laplacian random noise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Adult dataset 
dataset = pd.read_csv("adult.data.txt",
    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',na_values="?")
dataset.tail()

# Set parameters for Laplace function implementation
location = 1.0
scale = 1.0/0.125

#Find actual data count
datacount = dataset["Age"].value_counts()
print("datacount:\n",datacount)

# Gets random laplacian noise for all values
Laplacian_noise = np.random.laplace(location,scale, len(datacount))
print("Laplacian_noise:\n",Laplacian_noise)

# Add random noise generated from Laplace function to actual count
noisydata = datacount + Laplacian_noise
print("noisydata:\n",noisydata)

# 获取datacount索引字段，即国家名称
index=list(datacount.index)
# 构造series格式的噪声
laplacenoise=pd.Series(Laplacian_noise,index=index)
print("laplacenoise:\n",laplacenoise)
# Generate noisy histogram
# noisydata.plot(kind="bar",color = 'g')
# plt.show()
# print(list(datacount.index))
print(type(noisydata))
print(type(Laplacian_noise))
# print(type(datacount))
# x=range(len(datacount))
plt.ylabel("num")
plt.xlabel("")
plt.bar(range(len(noisydata)),datacount,label='datacount',fc='y')
plt.bar(range(len(noisydata)),laplacenoise,bottom=datacount,label='noise',tick_label=index,fc='r')
# l1=plt.bar(range(len(noisydata)),noisydata,color='b')
# l2=plt.bar(range(len(noisydata)),datacount,color='g')
plt.legend()
plt.show()