import matplotlib.pyplot as plt
import pandas as pd
from regression_function import *
import numpy as np


df = pd.read_csv('https://raw.githubusercontent.com/abhishekraghavan1/Linear-regression/master/headbrain.csv')
df = df[(df['Gender'] == 1) & (df['Age Range'] == 1)]
df = df[['Head Size(cm^3)','Brain Weight(grams)']]
df = df.sort_values(by=['Head Size(cm^3)'])

alp = Linear_Regression(df['Head Size(cm^3)'],df['Brain Weight(grams)'])

print("Simple Linear Regression Results")
print("--------------------------------")
print("Beta1: ", round(alp[1],3))
print("Beta2: ", round(alp[2],3))


difference = (df['Brain Weight(grams)'] - alp[0])**2
err = np.sqrt(difference.sum() / len(df))

print("Standard error: ", round(err,3))
print("Linear Regression Line: {} + {} * x".format(round(min(alp[0]),3), round(alp[2],3)))
print("Upper 0.95 interval line (+2std): {} + {} * x".format(round(min(alp[0]) + 1.96*err,3), round(alp[2],3)))
print("Lower 0.95 interval line (-2std): {} + {} * x".format(round(min(alp[0]) - 1.96*err,3), round(alp[2],3)))
print("---------------------------------------------")
df.plot(x='Head Size(cm^3)',y='Brain Weight(grams)')
plt.plot(df['Head Size(cm^3)'], alp[0]) 
plt.plot(df['Head Size(cm^3)'], alp[0] + 1.96*err)
plt.plot(df['Head Size(cm^3)'], alp[0] - 1.96*err)
plt.show()