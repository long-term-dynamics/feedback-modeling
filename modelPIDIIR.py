#! /usr/bin/env python

#
# Script name: modelPIDIIR.py
# Copyright 2021 Neal Patwari
#
# Version History:
#   Version 2.2:  Changed plot to x(n) rather than ratio. 18 January 2022. 
#	Version 2.0:  File created to add IIR filter library. 23 December 2021. 
#
# License: see LICENSE.md

import numpy as np
import numpy.linalg as linalg
# https://pykalman.github.io/
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


def estPIDModelIIR(x, a, setpoint=0):

	rows = x.shape[0]
	xp = x - setpoint         # The inequity at each time
	# The slope of the inequity xdot(n) = x(n)-x(n-1).
	# Note the 0 element of xdot is really at time 1
	xdot = xp[1:] - xp[0:-1] 

	# The cumulative inequity since time 0.
	# Note the 0 element of xdot is really at time 0
	xsum = IIRSum(xp, a)

	# The change in slope. Note the 0 element of xdotdot is really at time 1
	xdotdot = xdot[1:] - xdot[0:-1] # xdotdot(n) = x(n+1)-x(n)

	# We only really have rows-2 data points, we can't estimate xdot, xdotdot 
	# of the first year or xdotdot of the last year.
	S = np.zeros([rows-2, 3])
	S[:,0] = xp[1:-1]
	S[:,1] = xdot[0:-1]
	S[:,2] = xsum[1:-1]
	pseudoinverseS = linalg.pinv(S)

	model = pseudoinverseS.dot(xdotdot)
	return model


def IIRSum(xp, a):
	retsum = np.zeros(len(xp))
	retsum[0] = xp[0]
	for i in range(1,len(retsum)):
		retsum[i] = xp[i] + a * retsum[i-1]

	return retsum


def estimateEndingStateIIR(k_est, x, a):

	# A is the linear model for how the state progresses from year to year.
	A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 0, a]])

	# C is the observation matrix
	C = np.array([1, 0, 0])	

    # init Kalman filter and use it to estimate
	kf= KalmanFilter(A, C)

	kf.em(x, n_iter=5)
	(filtered_state_means, filtered_state_covs) = kf.filter(x)

	return filtered_state_means[-1,:]

	# Try an alternative IIR filter
def estimatePIDModelIIR(x, a, setpoint=0):

	k_est = estPIDModelIIR(x, a, setpoint)
	# print(k_est) #turn off printing for now

	for i in range(9):

		# A is the linear model for how the state progresses from year to year.
		A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 0, a]])

		# C is the observation matrix
		C = np.array([1, 0, 0])	
		# init Kalman filter and use it to estimate
		kf= KalmanFilter(A, C)

		kf.em(x, n_iter=5)
		(smoothed_state_means, smoothed_state_covs) = kf.smooth(x)

		k_est = estPIDModelIIR(smoothed_state_means[:,0], a, setpoint)
		# print(k_est)  #turn off printing for now

	return k_est

# Simulate the estimated model for a specified period of time
def simulatePIDModelIIR(startYear, duration, k_est, x, a, init_x=None, init_slope=0, init_sum=0):

    # Initialize the first year of the simulation.  You could make different assumptions
    # about how the first simulation year should be initialized.  But we don't
    # have any data before year 0, so there might need to be a different method
    # for a start year of 0.
	if init_x is None:
		if startYear >= 0 and startYear < len(x):
			init_x = x[startYear]
		else: 
			init_x = 0

	init_S = np.array([init_x, init_slope, init_sum])  # Ensure it has three elements and is a np.array
	#print(init_S)

	# A is the linear model for how the state progresses from year to year.
	A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 0, a]])
	
	# initialize the simulation output
	simOut    = np.zeros((duration,3))  
	simOut[0,:] = init_S

	# Calculate the next year's state from the prior year
	for i in range(duration-1):
		simOut[i+1] = A.dot(simOut[i])

    # Also return an array of the year numbers
	timeSim   = np.arange(startYear, startYear + duration)
	return simOut, timeSim


def extrapolateViaModelIIR(x, N, a, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8, offset=0, titlestr=''):
	rows         = len(x)
	x = x - offset
	if (trainingRows <=0) or (trainingRows >= (rows-1)):
		trainingRows = int(rows/2)

	k_training   = estimatePIDModelIIR(x[0:trainingRows], a, setpoint)

	# Simulate the estimated model for a specified period of time
	simDuration  = rows-trainingRows

	xhat         = estimateEndingStateIIR(k_training, x[0:trainingRows], a)

	# initalize the slope with the average slope over the first half.
	[simOut_test, timeSim0] = simulatePIDModelIIR(trainingRows, simDuration, k_training, x, a, \
		xhat[0], xhat[1], xhat[2])

    # Calculate the squared error for each prediction.
	err = np.square(simOut_test[:,0] - x[trainingRows:])
	# Calculate the mean absolute error over all years predicted.
	mae = np.sum(np.sqrt(err))/len(err)
    # Calculate the root mean squared error (RMSE) over all years predicted.
	rmse = np.sqrt(np.sum(err)/len(err))

	print("[kp kd ki] = " + str(k_training))
	print("RMSE = " + str(rmse))

	#plt.clf()
	plt.plot(N[0:trainingRows], x[0:trainingRows] + offset, 'g-o', linewidth=2, label='Actual Training')
	plt.plot(N[trainingRows:], simOut_test[:,0] + offset, 'b-', linewidth=1, label='Predicted Test')
	plt.plot(N[trainingRows:], x[trainingRows:] + offset, 'g:', linewidth=2, label='Actual Test')
	plt.plot([min(N)-0.5, max(N)+0.5], [offset, offset], 'k-', linewidth=1)
	plt.xlim(min(N)-0.5, max(N)+0.5)
	plt.title(titlestr)
	plt.xlabel("Year", fontsize=20)
	plt.ylabel(ylabelstr, fontsize=16)
	plt.xticks(range(int(min(N)), int(max(N)), tickDelta))
	plt.legend(fontsize=16)
	plt.grid(True)
	plt.show()

	return simOut_test, err, rmse, mae, k_training