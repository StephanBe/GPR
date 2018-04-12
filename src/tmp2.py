#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:55:58 2018

@author: stephan
"""

import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
    
def rotate(a, newXAxis):
    r = newXAxis
    normX = r / sqrt(np.dot(r.T,r))
    normY = [-normX[1], normX[0]]
    b = np.dot(np.array([normX, normY]).T, a)
    return(b)

"""return true if v > 1 km/h or any speed given"""
def isMoving(deltaXPosition, deltaYPosition, deltaTime, fasterThankmh=1.0):
    x = deltaXPosition
    y = deltaYPosition
    t = deltaTime
    if t*t == 0.:
        return False
    if hasattr(x, "__len__"):
        x = x[0]
    if hasattr(y, "__len__"):
        y = y[0]
    if hasattr(t, "__len__"):
        t = t[0]
    speed = float(fasterThankmh)
    return((x*x + y*y) / (t*t) > 0.077160*speed*speed)

def velocity_verlet_integration(Xacc, Yacc,
                                x0=0., y0=0.,
                                vx_0=0, vy_0=0,
                                forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    x[0] = x0
    y[0] = y0
    vx[0] = vx_0
    vy[0] = vy_0
    for i in range(len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,:], forward)
        x[i+1] = x[i] + vx[i]*dt + 1.0/2.0*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1.0/2.0*a[1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
    return x, y


"""test circle"""
centripetal=-0.2
N = 0.01
xCircle = np.array(range(int(100*10**N)))/float(10**N)
yCircle = np.array([[0.0, centripetal] for i in xCircle])
xvvi, yvvi = velocity_verlet_integration(xCircle, yCircle, 0., 0., 1., 0.)
#plot it
plt.plot(xvvi, yvvi, ".-", label='position with "velocity verlet" integration')