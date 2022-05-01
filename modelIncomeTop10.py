#! /usr/bin/env python

#
# Script name: modelPayGap.py
# Copyright 2021 Neal Patwari
#
# Version History:
#   Version 2.2:  Changed plot to x(n) rather than ratio. 18 January 2022. 
#   Version 2.0:  Added IIR Filter. 23 December 2021. 
#   Version 1.0:  Initial Release.  30 April 2021.
#
# License: see LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import modelPIDIIR as mpi


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


fname    = 'data/PicketySaezIncomeTopDecile.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1))
# Use data only post WW2.
year0ind = 28
N        = data_in[year0ind:, 0]  # This data set is in forward time order.
x        = data_in[year0ind:, 1] 
rows     = len(x)
offset   = 1
x        = (x / 10.0) - offset # The ideal would be 10%, so the "inequality" is 10

ylabel   = "x(n), Income of Top 10%"

plt.ion()
alpha = 0.99

# use first half of data
plt.figure(1)
trainingRows = int(rows/2.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 10)
plt.legend(loc='upper left', fontsize=14)
plt.ylim(offset - 1.05, offset + 4)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_incomeTop10_1982-2018_IIR.eps")
plt.savefig("extrapolate_incomeTop10_1982-2018_IIR.png")

# use first 2/3rds of data
plt.figure(2)
trainingRows = int(2.0*rows/3.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 10)
plt.legend(loc='upper left', fontsize=14)
plt.ylim(offset - 1.05, offset + 4)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_incomeTop10_1994-2018_IIR.eps")
plt.savefig("extrapolate_incomeTop10_1994-2018_IIR.png")


###################
# Estimate k parameters
###################

# first half
k_training   = mpi.estimatePIDModelIIR(x[:int(rows/2.0)], alpha, 0.0)
print("First half: ", k_training)

#2/3rds
k_training   = mpi.estimatePIDModelIIR(x[:int(2.0*rows/3.0)], alpha, 0.0)
print("Two Thirds: ", k_training)

# all data
k_training   = mpi.estimatePIDModelIIR(x, alpha, 0.0)
print("All Data: ", k_training)

# Estimate parameters from 2nd half of data
k_training   = mpi.estimatePIDModelIIR(x[int(rows/2.0):], alpha, 0.0)
print("Second Half: ", k_training)
