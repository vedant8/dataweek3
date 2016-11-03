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
x1 = np.linspace(15.0, 30.0, 750.0)
y1 = np.linspace(950.0, 1050.0, 750.0)
l1 =np.ndarray(shape=(750,750),dtype=float)
X, Y = np.meshgrid(x1, y1)
# for i in range(0,750,1):
#     for j in range(0,750,1):
#         d=[x1[i],y1[j],results.x[2]]
#         m=line(d)
#
#         if abs(m-v1)<=0.1:
#             l1[i, j]=1
#         elif abs(m-v2)<=0.1:
#             l1[i, j]=1
a = np.sum(temperature)
t1 = np.square(temperature)
b = np.sum(t1)
c = np.sum(length)
t2 = np.multiply(temperature,length)
d = np.sum(t2)
t3 = np.square(length)
e = np.sum(t3)
c1=results.x[0]
c2=results.x[1]
F = ((X**2)*b+(50*(Y**2))+(e)+(2*X*Y*a)-(2*Y*c)-(2*X*d))
s=0
for i in range (0,50,1):
   s+= (length[i]-(results.x[0]*temperature[i]+results.x[1]))**2
G1 = (2 * (results.x[2]**2)) + s
G2 = (4 * 2 * (results.x[2]**2)) + s
#errorinterval on m
coeff=[]
coeff.append(b)
coeff.append(2*a*c2-2*d)
coeff.append(50*c2**2-2*c*c2+e-s-results.x[2])
m12 = np.roots(coeff)
m1 = min(m12[0],m12[1])
m2 = max(m12[0],m12[1])
#error interval in c
coeff=[]
coeff.append(50)
coeff.append(2*a*c1-2*c)
coeff.append(-2*d*c1+b*c1**2+e-s-results.x[2])
print ("Error interval for m is [%5.2f,%5.2f]")%(m1,m2)
i12 = np.roots(coeff)
i1 = min(i12[0],i12[1])
i2 = max(i12[0],i12[1])
print ("Error interval for c is [%5.2f,%5.2f]")%(i1,i2)
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

