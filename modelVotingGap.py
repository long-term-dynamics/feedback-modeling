#! /usr/bin/env python

#
# Script name: modelVotingGap.py
# Copyright 2021 Neal Patwari
#
# Version History:
#   Version 2.2:  Changed plot to x(n) rather than ratio. 18 January 2022. 
#   Version 2.0:  Added IIR filter. 23 December 2021. 
#   Version 1.0:  Initial Release.  30 April 2021.
#
# License: see LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import modelPIDIIR as mpi


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

fname    = 'data/BlackWhiteVotingPercentageUS.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1,2,3))
rows     = data_in.shape[0]

# Keep track of the earliest year; make it year 0; each row is two years.
year0    = data_in[-1,0]    # Last year in the data set
N        = np.flip(data_in[0:, 0]) 

# Ratio minus offset (equality)
offset   = 1
x        = np.flip(data_in[0:, 2] / data_in[0:,3]) - offset
ylabel   = "x(n), Voting Gap White vs. Black"

plt.ion()
alpha = 0.9

# first half of data
plt.figure(1)
trainingRows = int(rows/2.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.8 - offset, 1.5 - offset)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_votinggap_1992-2018_IIR.eps")
plt.savefig("extrapolate_votinggap_1992-2018_IIR.png")

# first 2/3rds of data
plt.figure(2)
trainingRows = int(2.0*rows/3.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.8 - offset, 1.5 - offset)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_votinggap_2000-2018_IIR.eps")
plt.savefig("extrapolate_votinggap_2000-2018_IIR.png")

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

