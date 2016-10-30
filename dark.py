import numpy as np
from array import *
import math
import scipy
from scipy.stats import binom
import csv
import matplotlib.pyplot as plt
class plotting:
    def panda(self):
        temperature = [0.0] * 0
        length = [0.0] * 0
        background=np.zeros(40)
        signal=np.zeros(40)
        m=20*0.1
        with open('recoilenergydata_EP219.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                temperature.append(float(row[0]))
                length.append(float(row[1]))
        for i in range (0,40,1):
         background[i]=-temperature[i]/10
         if ((temperature[i]<5)or (temperature[i]>25)):
             signal[i]=0
         elif((temperature[i]>5) and (temperature[i]<15)):
             signal[i]=m*(temperature[i]-5)
         elif((temperature[i]>15) or (temperature[i]<25)):
             signal[i]=m*(25-temperature[i])

        reading=signal+1000*np.exp(background)
        w = abs(temperature[1]) - abs(temperature[0])
        plt.bar(temperature, length, width=w, alpha=0.5, align='center')
        plt.bar(temperature,reading,width=w, alpha=0.5, align='center' )

        plt.title("Histogram for Probability Density of z")
        plt.ylabel("Probability Density")
        plt.xlabel("z")
        plt.legend()
        plt.show()
plotting().panda()