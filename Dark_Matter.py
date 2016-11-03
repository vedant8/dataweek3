import numpy as np
from scipy.optimize import minimize
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

# Deciding the width of the histogram
w = abs(energy[1]) - abs(energy[0])

# Plotting the Histogram of the given data
plt.bar(energy, events, width=w, alpha=0.5, align='center')
plt.title("Histogram of the Events Measured")
plt.xlabel("Energy")
plt.ylabel("Frequency of the events measured")
plt.show()

# Plotting the Histogram of the mean background events
plt.bar(energy, 1000 * np.exp(background), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the mean background events")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean background events")
plt.show()

# Function to calculate mean reading events for a given value of sigma
def readingMean(sigma):
    for i in range(len(energy)):
        if (energy[i] < 5) or (energy[i] > 25):
            signal[i] = 0
        elif (energy[i] > 5) and (energy[i] < 15):
            signal[i] = sigma * 20 * (energy[i] - 5)
        elif (energy[i] > 15) or (energy[i] < 25):
            signal[i] = sigma * 20 * (25 - energy[i])
    reading = signal + 1000 * np.exp(background)
    return reading

# Plotting histogram for different value of sigma
plt.bar(energy, readingMean(0.01), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the Mean Events with sigma = 0.01")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean events")
plt.show()

plt.bar(energy, readingMean(0.1), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the Mean Events with sigma = 0.1")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean events")
plt.show()

plt.bar(energy, readingMean(1), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the Mean Events with sigma = 1")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean events")
plt.show()

plt.bar(energy, readingMean(10), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the Mean Events with sigma = 10")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean events")
plt.show()

plt.bar(energy, readingMean(100), width=w, color='red', alpha=0.5, align='center')
plt.title("Histogram of the Mean Events with sigma = 100")
plt.xlabel("Energy")
plt.ylabel("Frequency of the mean events")
plt.show()


#################################################################################################
#                              Log Likelihood Function and Calculation                          #
#################################################################################################

# Calculates the negative of the log likelihood function
def line(param, d, energy):
    s = param
    read = readingMean(s)
    # Calculate negative log likelihood
    LL = -np.sum(d * np.log(read) - read)
    return LL

# Plotting the log likelihood function between 0 to 100
sigma = np.arange(0, 10, 0.1)
for i in range(len(sigma)):
    plt.plot(sigma[i], -line(sigma[i], events, energy), 'ro')
plt.title("Log Likelihood function for the theory parameter sigma")
plt.xlabel("Theory parameter sigma")
plt.ylabel("Log Likelihood Value")
plt.show()

# Calculating the maximum Log likelihood
init = np.array([0.5])
results = minimize(line, init, args=(events, energy), method='Nelder-Mead')      # minimization
print "The sigma at which Log Likelihood function is maximum is at %s"%results.x  # best fit parameter

# Calculating the 1sigma interval
LLmin = -line(results.x, events, energy)
OneSigma = np.array([0.0, 0.0])
LLOneSigma = float('inf')
# Calculating the lower bound
for i in range(0, int(10 ** 5 * results.x)):
    sigma = float(i) / 10 ** 5
    x = -line(sigma, events, energy)
    if abs(x - (LLmin - 0.5)) < LLOneSigma:
        LLOneSigma = abs(x - (LLmin - 0.5))
        OneSigma[0] = sigma
# Calculating the upper bound
y = []
LLOneSigma = float('inf')
for i in range(int(10 ** 5 * results.x), 10 ** 5):
    sigma = float(i) / 10 ** 5
    x = -line(sigma, events, energy)
    if abs(x - (LLmin - 0.5)) < LLOneSigma:
        LLOneSigma = abs(x - (LLmin - 0.5))
        OneSigma[1] = sigma

print "The 1 sigma interval is ( %s, %s )" %(OneSigma[0], OneSigma[1])

