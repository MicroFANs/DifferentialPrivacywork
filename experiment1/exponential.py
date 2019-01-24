""" Program to add noise generated from Exponential mechanism
    to Original Adult dataset.
"""   
# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Adult dataset
dataset = pd.read_csv("adult.data.txt", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',na_values="?")
dataset.tail()

datacount = dataset["Country"].value_counts()
print("originaldata:")
print(datacount)
# Generate random noise from exponential function.
Exponential_noise = np.random.exponential(1)     # Keep max limit = 1

print ("Exponentially generated noise:", Exponential_noise)

"""Add random noise drawn from Exponential function to Original data count"""
noisydata = datacount + Exponential_noise
print("noisydata:")
print(noisydata)
#Plot histogram for Noisy data
noisydata.plot(kind="bar", color = 'r')
plt.hist(noisydata)
plt.show()