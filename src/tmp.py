#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:53:59 2018

@author: stephan
"""

"""
trajectory.py
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

def gauss(x, m, s):
    return 1/(s*sqrt(2*pi)) * np.exp(-1./2. * ((x-m)/s)**2)


x = np.array(range(-1000, 1000))/100
plt.figure()
plt.plot(x, gauss(x, 0, 0.5), label='$\sigma = 0.5$')
plt.plot(x, gauss(x, 0, 1), label='$\sigma = 1$')
plt.plot(x, gauss(x, 1, 2), label='$\sigma = 2$')
plt.legend()







# =============================================================================
# 
# from scipy.integrate import odeint
# import Data as data
# 
# ## set initial conditions and parameters
# g = 9.81            # acceleration due to gravity
# th = 45.            # set launch angle
# th = th * np.pi/180.   # convert launch angle to radians
# v0 = 10.0           # set speed
# 
# x0=0                # specify initial conditions
# y0=0
# vx0 = v0*np.cos(th)
# vy0 = v0*np.sin(th)
# 
# ## define function to compute f(X,t)
# def f_func(state,time):
#     t = np.arange(len(data.Xacc))[data.Xacc == time]
#     f = np.zeros(4)    # create array to hold f vector
#     f[0] = state[state[0]] # f[0] = x component of velocity
#     f[1] = state[state[1]] # f[1] = x component of velocity
#     f[2] = data.Yacc[time,1]        # f[2] = acceleration in x direction
#     f[3] = data.Yacc[time,2]       # f[3] = acceleration in y direction
#     return f
# 
# ## set initial state vector and time array
# X0 = [ x0, y0, vx0, vy0]        # set initial state of the system
# t0 = 0.
# tf_str = 30
# tf = float(tf_str)
# 
# # create time array starting at t0, ending at tf with a spacing tau
# t = data.Xacc
# 
# ## solve ODE using odeint
# X = odeint(f_func,X0,t) # returns an 2-dimensional array with the
#                         # first index specifying the time and the
#                         # second index specifying the component of
#                         # the state vector
# 
# # putting ':' as an index specifies all of the elements for
# # that index so x, y, vx, and vy are arrays at times specified
# # in the time array
# x = X[:,0]
# y = X[:,1]
# vx = X[:,2]
# vy = X[:,3]
# 
# ## plot the trajectory
# plt.figure(1)
# plt.clf()
# 
# plt.plot(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# =============================================================================
