import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv
import xlrd
from scipy.optimize import minimize
import pylab as py

#input data from csv file
temperature = [0.0]*0
length = [0.0]*0

with open('linearexpansion.csv') as csvfile:
     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in reader:
         temperature.append(float(row[0]))
         length.append(float(row[1]))
#function to retrun log likelihood given array of size 3
def line(params):
    m = params[0]#slope parameter
    y0 = params[1]#intercept
    sd = params[2]#sd, set to 0.5

    yPred=[]
    for i in range(len(temperature)):
        yPred.append(y0+m*temperature[i])
    # Calculate negative log likelihood
    LL = -np.sum(stats.norm.logpdf(length, loc=yPred, scale=sd ) )
    return(LL)


initParams = [1.0, 1000.0, 0.5]

results = minimize(line, initParams, method='Nelder-Mead')#minimization
print results.x#best fit params
#1 and 2 sigma part...
v1=line((results.x))+1
v2=line((results.x))+2
x1 = np.linspace(20.0, 25.0, 5000.0)
y1 = np.linspace(950.0, 1100.0, 5000.0)
l1x =[]
l1y =[]
for i in range(0,5000,1):
    d=[x1[i],y1[i],results.x[2]]
    m=line(d)
    if(abs(m-v1)<=0.5):
        l1x.append(x1[i])
        l1y.append(y1[i])


X, Y = np.meshgrid(l1x, l1y)
plt.figure()
plt.contour(X, Y)#contour map
plt.colorbar()
plt.title('Contour Map of Probability Distribution Function of x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
