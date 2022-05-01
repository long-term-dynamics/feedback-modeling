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


fname    = 'data/GenderPayGapData.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=2, usecols=(0,1))
rows     = data_in.shape[0]
# Flip is because I know this data set is in reverse time order.
N        = np.flip(data_in[0:, 0])  + 1960
x        = 1.0/(1.0 - np.flip(data_in[0:, 1]) ) - 1

ylabel   = "x(n), Pay Gap Men vs. Women"



plt.ion()

alpha = 0.9

# train on first half
plt.figure(1)
trainingRows = int(rows/2.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(-0.1, 0.8)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_genderpaygap_1990-2018_IIR.eps")
plt.savefig("extrapolate_genderpaygap_1990-2018_IIR.png")

# train on first two thirds
plt.figure(2)
trainingRows = int(2.0*rows/3.0)
mpi.extrapolateViaModelIIR(x, N, alpha, trainingRows, 0.0, ylabel, 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(-0.1, 0.8)
plt.tight_layout()
plt.show()
plt.savefig("extrapolate_genderpaygap_1999-2018_IIR_thirds.eps")
plt.savefig("extrapolate_genderpaygap_1999-2018_IIR_thirds.png")

## Estimate k parameters
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