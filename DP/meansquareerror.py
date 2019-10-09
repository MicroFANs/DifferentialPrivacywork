# Program to compute Mean Squared Error(MSE) and show plot between Epsilon and MSE
# in Laplace Mechanism
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load the Adult dataset 
dataset = pd.read_csv("adult.data.txt", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',na_values="?")
dataset.tail()

# Laplace function implementation, takes epsilon as an argument
def Laplacian_func(eps):     
 x = 0.01
 mu = 0                    # mean
 return ((eps/2.0) * np.exp(-abs(x - mu)*eps))
 
datacount = dataset["Race"].value_counts()    #Store actual data count
tmp = []
mselist = []
fig=plt.figure()
print(datacount)
# Call laplace for all values of epsilon, calculate MSE for each case and plot.
noise = Laplacian_func(1/0.125)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount- noisydata)**2).mean(axis=0)  
mselist.append(mse)
noise = Laplacian_func(1/0.114)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0) 
mselist.append(mse)
noise = Laplacian_func(1/0.097)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)  
mselist.append(mse)
noise = Laplacian_func(1/0.08)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0) 
mselist.append(mse)
noise = Laplacian_func(1/0.07099)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1/0.07255)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1/0.0847)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1/0.1028)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1/0.119)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1/0.129)
noisydata = datacount + noise
print("*",noise)
print(noisydata,'\n')
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
print(mselist)
for i in range(0,50):
  for j in list(mselist):
      tmp.append(j)      
print ("Average Mean Square Error is- ") 
print (np.average(tmp))   

epsval = [1.0] 
x = 1.0
for i in range(1,10):
   x -= 0.1
   epsval.append(x)   
ax = fig.add_subplot(111)
ax.plot(epsval,mselist)
#plt.xlabel('Epsilon')
plt.ylabel('MSE')
plt.show()

