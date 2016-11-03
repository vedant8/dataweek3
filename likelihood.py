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
x1 = np.linspace(20.0, 26.0, 750.0)
y1 = np.linspace(970.0, 1020.0, 750.0)
l1 =np.ndarray(shape=(750,750),dtype=float)
X, Y = np.meshgrid(x1, y1)
sumX = np.sum(temperature)
sumX2 = np.sum(np.square(temperature))
sumY = np.sum(length)
sumXY = np.sum(np.multiply(temperature,length))
sumY2 = np.sum(np.square(length))
m=results.x[0]
c=results.x[1]
F = ((X**2)*sumX2+(50*(Y**2))+(2*X*Y*sumX)-(2*Y*sumY)-(2*X*sumXY) + sumY2)
sum = 0
for i in range (0,50,1):
   sum += (length[i]-(m*temperature[i]+c))**2
G1 = (2 * (results.x[2]**2)) + sum
G2 = (2 * 2 * (results.x[2]**2)) + sum

#seting up a symmertrical meshgrid
plt.figure()
plot1 = plt.contour(X, Y,(F-G1),[0],colors ='y')#contour map
plot2 = plt.contour(X,Y,(F-G2),[0])
labels=['1-Sigma Region','2-Sigma Region']
plot1.collections[0].set_label(labels[0])
plot2.collections[0].set_label(labels[1])

plt.legend(loc = 'upper right')
plt.title('Contour Map of error regions')
plt.xlabel('Coefficient of Linear Expansion')
plt.ylabel('Length')
plt.show()

#errorinterval on m
coeff=[]
coeff.append(sumX2)
coeff.append(2*c*sumX - 2*sumXY)
coeff.append(50*c**2 - 2*c*sumY + sumY2 - sum - 2*results.x[2]**2)
m12 = np.roots(coeff)
m1 = min(m12[0],m12[1])
m2 = max(m12[0],m12[1])
#error interval in c
coeff=[]
coeff.append(50)
coeff.append(2*m*sumX-2*sumY)
coeff.append(-2*m*sumXY + sumX2*m**2 + sumY2 - sum - 2*results.x[2]**2)
print ("Error interval for m is [%5.2f,%5.2f]")%(m1,m2)
c12 = np.roots(coeff)
c1 = min(c12[0],c12[1])
c2 = max(c12[0],c12[1])
print ("Error interval for c is [%5.2f,%5.2f]")%(c1,c2)


