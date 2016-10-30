import numpy as np
from array import *
import math
import scipy
from scipy.stats import binom
import csv
import matplotlib.pyplot as plt
class plotting:
    def panda(self):
        energy = [0.0] * 0
        events = [0.0] * 0
        background=np.zeros(40)
        signal=np.zeros(40)
        m=20*0.1
        with open('recoilenergydata_EP219.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                energy.append(float(row[0]))
                events.append(float(row[1]))
        for i in range (0,40,1):
         background[i]=-energy[i]/10
         if ((energy[i]<5)or (energy[i]>25)):
             signal[i]=0
         elif((energy[i]>5) and (energy[i]<15)):
             signal[i]=m*(energy[i]-5)
         elif((energy[i]>15) or (energy[i]<25)):
             signal[i]=m*(25-energy[i])

        reading=signal+1000*np.exp(background)
        w = abs(energy[1]) - abs(energy[0])
        plt.bar(energy, events, width=w, alpha=0.5, align='center')
        plt.bar(energy,reading,width=w, alpha=0.5, align='center' )

        plt.title("Histogram for Probability Density of z")
        plt.ylabel("Probability Density")
        plt.xlabel("z")
        plt.legend()
        plt.show()
plotting().panda()