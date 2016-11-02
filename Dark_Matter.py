import numpy as np
from scipy.optimize import minimize
from array import *
import math
import scipy
from scipy.stats import binom
import csv
import matplotlib.pyplot as plt

energy = [0.0] * 0
events = [0.0] * 0
signal = np.zeros(40)
m = 20 * 0.1766

# Obtaining the data
fhand = open('recoilenergydata_EP219.csv')
for line in fhand:
    if line[0] == '#':
        continue
    p = line.split(',')
    energy.append(float(p[0]))
    events.append(float(p[1]))
    background = np.divide(np.multiply(energy, -1), 10)

w = abs(energy[1]) - abs(energy[0])

plt.bar(energy, events, width=w, alpha=0.5, align='center')
plt.show()

plt.bar(energy, 1000 * np.exp(background), width=w, color='red', alpha=0.5, align='center')
plt.show()

def signalMean (sigma):
    for i in range(len(energy)):
        if (energy[i] < 5) or (energy[i] > 25):
            signal[i] = 0
        elif (energy[i] > 5) and (energy[i] < 15):
            signal[i] = sigma* 20 * (energy[i] - 5)
        elif (energy[i] > 15) or (energy[i] < 25):
            signal[i] = sigma * 20 * (25 - energy[i])
    reading = signal + 1000 * np.exp(background)
    return reading

plt.bar(energy, signalMean(0.01), width=w, color='red', alpha=0.5, align='center')
plt.show()
plt.bar(energy, signalMean(0.1), width=w, color='red', alpha=0.5, align='center')
plt.show()
plt.bar(energy, signalMean(1), width=w, color='red', alpha=0.5, align='center')
plt.show()
plt.bar(energy, signalMean(10), width=w, color='red', alpha=0.5, align='center')
plt.show()
plt.bar(energy, signalMean(100), width=w, color='red', alpha=0.5, align='center')
plt.show()


#################################################################################################

def line(param, d, energy):
    s = param
    yPred = []
    xPred = np.divide(energy, 10)
    for i in range(len(energy)):
        if (energy[i] < 5) or (energy[i] > 25):
            yPred.append(0)
        elif (energy[i] > 5) and (energy[i] < 15):
            yPred.append(s * 20 * (energy[i] - 5))
        elif (energy[i] > 15) or (energy[i] < 25):
            yPred.append(s * 20 * (25 - energy[i]))
    read = yPred + np.multiply(1000, np.exp(-xPred))
    # Calculate negative log likelihood
    LL = -np.sum(d * np.log(read) - read)
    return LL

sigma = np.arange(0, 1, 0.01)
for i in range(len(sigma)):
    plt.plot(sigma[i], line(sigma[i], events, energy),'ro')
plt.show()


init = np.array([0.5])
# x1 = np.linspace(0.0, 10.0, 5000.0
# plt.plot(x1,line(x1,events,energy)
results = minimize(line, init, args=(events, energy), method='Nelder-Mead')  # minimization
print results.x  # best fit params
LLmin = line(results.x,events,energy)
print LLmin
OneSigma = np.array([0,0])
k = 0
x = []

for i in range(0, int(10**5*LLmin)):
    sigma = float(i)/10**5
    x.append(line(sigma, events, energy))

OneSigma[0] = min(x, key=lambda x: abs(x-(LLmin+0.5)))

y = []

for i in range(int(10**5*LLmin),10**5):
    sigma = float(i)/10**5
    y.append(line(sigma, events, energy))

OneSigma[1] = min(x, key=lambda x: abs(x-(LLmin+0.5)))

print OneSigma


