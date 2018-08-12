# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:05:28 2018

@author: Stephan

http://www.michaelkeutel.de/blog/rotation-matrices-vector-basis/
"""

#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import FancyArrowPatch
#from math import cos
#from math import sqrt
from mpl_toolkits.mplot3d import proj3d #wird benutzt! Warnung ignorieren
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from scipy import integrate
from matplotlib.widgets import Slider
from math import sqrt, pi, atan2
from bisect import bisect_left, bisect_right

import Data
from vectorRotation import rotate

def gpr(x_train, y_train, x_pred):
    #WhiteKernel for noise estimation (alternatively set alpha in GaussianProcessRegressor()) 
    #ConstantKernel for signal variance
    #RBF for length-scale
    kernel = RBF(0.1, (0.01, 10))*ConstantKernel(1.0, (0.1, 100)) + WhiteKernel(0.1, (0.01,10))
    #noise = 0.1
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
    mean = np.mean(y_train, 0)
    gp.fit(x_train, y_train-mean)
    #print(gp.kernel_)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    y_pred += mean
    return y_pred, sigma


"""data"""
Xacc = Data.Xacc_corrected2.flatten()
Yacc = Data.Yacc
Xgps = Data.Xgps
Ygps = Data.Ygps
Xgyr = Data.Xgyr_corrected2
Ygyr = Data.Ygyr
#=============================================================================
# tmp = np.copy(Yacc)
# #vorher: android x (Yacc[:,0] = nach unten beschleunigen)
# DOWN = 0
# Yacc[:,DOWN] = tmp[:,0] #danach: Yacc[:,0] = beschleunigung nach unten
# #vorher: android z (Yacc[:,2] = nach hinten beschleunigen)
# FORWARD = 1
# Yacc[:,FORWARD] = -tmp[:,2] #danach: Yacc[:,1] = beschleunigung nach vorn
# #vorher: android y (Yacc[:,1] = nach links beschleunigen)
# LEFT = 2
# Yacc[:,LEFT] = tmp[:,1] #danach: Yacc[:,2] = beschleunigung nach links
# #Yacc[:,LEFT] = np.zeros(Yacc[:,LEFT].shape)
#=============================================================================

def isMoving(deltaXPosition, deltaYPosition, deltaTime=1., fasterThankmh=0.1):
    """return true if v > 1 km/h or any speed given"""
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

# =============================================================================
# def get_rotation(time, acceleration, initial_velocity):
#     velocity = np.zeros(acceleration.shape)
#     velocity[0,:] = initial_velocity
#     for i in range(len(acceleration)-1):
#         f  = velocity[i,:]
#         a  = acceleration[i,:]
#         v  = velocity[i,:] 
#         dt = time[i+1] - time[i]
#         change = 1.
#         for it in range(1):
#             if change < 1e-6:
#                 break
#             before = f
#             R = np.array([[f[0], -f[1]],
#                           [f[1],  f[0]]]) * 1 / sqrt(f.T @ f)
#             f = R @ a * dt + v
#             change = np.sum(np.abs(before - f))
#             print(f)
#             #print("current change "+str(change))
#         velocity[i+1,:] = f
#     return velocity
# =============================================================================


#https://en.wikipedia.org/wiki/Verlet_integration
def verlet_integration(Xacc, Yacc, x0=0., y0=0., vx_0=0., vy_0=0., forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    x[0],  y[0]  = x0,   y0
    vx[0], vy[0] = vx_0, vy_0
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = x[0] + vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = y[0] + vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    if isMoving(x[1]-x[0], y[1]-y[0], dt):
        forward = np.array([x[1]-x[0], y[1]-y[0]])
    for i in range(1, len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = 2*x[i] - x[i-1] + a[0]*dt*dt
        y[i+1] = 2*y[i] - y[i-1] + a[1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
    return x, y, vx, vy

#https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
def velocity_verlet_integration(Xacc, Yacc, x0=0., y0=0., vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
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
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = x[i] + vx[i]*dt + 1.0/2.0*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1.0/2.0*a[1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
    return x, y, vx, vy

def my_integration(t, a_object,
                   x0=0., y0=0.,
                   vx_0=0, vy_0=0,
                   forward=np.array([1.0, 0.0]),
                   return_rotated_acceleration=False):
    """
    Recursive approach to the orientation problem.
    
    t is the time stamp corresponding to a.
    
    columns of object space acceleration a: [DOWN, FORWARD, LEFT]
    
    Returns
    -------
    p_x, p_y, v_x, v_y, v[, a]
    """
    a_o = a_object[:,1:]
    v = np.zeros((len(t), 2))
    p = np.zeros((len(t), 2))
    p[0,:] = np.array([x0, y0])
    v[0,:] = np.array([vx_0, vy_0])
    v[1,:] = np.array([vx_0, vy_0])
    #if v[0,0] != 0. and v[0,1] != 0.:
    if isMoving(v[0,0], v[0,1], 1):
        forward = v[0,:]
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        a = rotate(a_o[i,:], forward)
        for j in range(10):
            aNext = rotate(a_o[i+1,:], forward)
            v[i+1,:] = v[i,:] + dt*(a + aNext)/2.
            #if v[i+1,0] != 0. and v[i+1,1] != 0.:
            if isMoving(v[i+1,0], v[i+1,1], 1):
                forward = v[i+1,:]
        if i < len(t)-2:
            v[i+2,:] = v[i+1,:]
        p[i+1,:] = p[i,:] + dt*v[i,:] + dt*dt*(a+aNext)/4.
    if return_rotated_acceleration:
        return p[:,0], p[:,1], v[:,0], v[:,1], v, a
    else:
        return p[:,0], p[:,1], v[:,0], v[:,1], v

def my_integrationold(t, a_object,
                      x0=0., y0=0.,
                      vx_0=0, vy_0=0,
                      forward=np.array([1.0, 0.0]),
                      return_rotated_acceleration=False):
    a_o = a_object[:,1:]
    v = np.zeros((len(t), 2))
    p = np.zeros((len(t), 2))
    a = np.zeros((len(t), 2))
    p[0,:] = np.array([x0, y0])
    v[0,:] = np.array([vx_0, vy_0])
    v[1,:] = np.array([vx_0, vy_0])
    a[0,:] = a_o[0,:]
    for i in range(len(t)-1):
        for j in range(10):
            dt = t[i+1]-t[i]
            a[i,:] = rotate(a_o[i,:],   v[i,:]+v[i+1,:])
            p[i+1,:] = p[i,:] + v[i,:]*dt + 1.0/2.0*a[i,:]*dt*dt
            a[i+1,:] = rotate(a_o[i+1,:], v[i,:]+v[i+1,:])
            v[i+1,:] = v[i,:] + dt*(a[i,:] + a[i+1,:])/2.
    if return_rotated_acceleration:
        return p[:,0], p[:,1], v[:,0], v[:,1], v, a
    else:
        return p[:,0], p[:,1], v[:,0], v[:,1], v

#my integration
def my_integration_oldest(t, Yacc,
                          x0=0., y0=0.,
                          vx_0=0., vy_0=0.,
                          forward=np.array([1.0, 0.0]),
                          return_rotated_acceleration=False):
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = vx_0
    vy[0] = vy_0
    tmp = forward
    forward = np.zeros((len(t), 2))
    forward[0:,:] = tmp
    #first iteration
    dt = Xacc[1]-Xacc[0]
    a = np.zeros((len(t), 2))
    a[0,:] = rotate(Yacc[0,1:], forward[0,:])
    x[1] = x[0] + vx[0]*dt + 1.0/2.0*a[0,0]*dt*dt
    y[1] = y[0] + vy[0]*dt + 1.0/2.0*a[0,1]*dt*dt
    if isMoving(x[1]-x[0], y[1]-y[0], dt):
        forward[1:,:] = np.array([x[1]-x[0], y[1]-y[0]])
    aNext = rotate(Yacc[1,1:], forward[1,:])
    #Integration durch trapezoidal rule
    vx[1] = vx[0] + dt*(a[0,0] + aNext[0])/2
    vy[1] = vy[0] + dt*(a[0,1] + aNext[1])/2
    for i in range(1, len(t)-1):
        dt = t[i+1]-t[i]
        a[i,:] = rotate(Yacc[i,1:], forward[i,:])
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[i,0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[i,1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward[(i+1):,:] = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward[i+1,:])
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[i,0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[i,1] + aNext[1])/2
        
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt) \
                and isMoving(x[i]-x[i-1], y[i]-y[i-1], t[i]-t[i-1]):
            #zentrale differenz für eine bessere Richtungsschätzung
            tb = t[i]   - t[i-1]
            ta = t[i+1] - t[i] # = dt
            forward[i:,0] = 1.0/(tb+ta)*\
                (tb/ta*x[i+1] + (ta**2 - tb**2)/(ta*tb)*x[i] - ta/tb*x[i-1])
            forward[i:,1] = 1.0/(tb+ta)*\
                (tb/ta*y[i+1] + (ta**2 - tb**2)/(ta*tb)*y[i] - ta/tb*y[i-1])
            #korrigiere den Schritt mit der besseren Richtungsschätzung
            a[i,:] = rotate(Yacc[i,1:], forward[i,:])
            x[i+1] = x[i] + vx[i]*dt + 1/2*a[i,0]*dt*dt
            y[i+1] = y[i] + vy[i]*dt + 1/2*a[i,1]*dt*dt
            #korrigiere die nächste Beschleunigung mit dem besseren Schritt
            forward[(i+1):,:] = np.array([x[i+1]-x[i], y[i+1]-y[i]])
            aNext = rotate(Yacc[i+1,1:], forward[i+1,:])
            #Integration durch trapezoidal rule
            vx[i+1] = vx[i] + dt*(a[i,0] + aNext[0])/2
            vy[i+1] = vy[i] + dt*(a[i,1] + aNext[1])/2
        
    if return_rotated_acceleration:
        return x, y, vx, vy, forward, a
    else:
        return x, y, vx, vy, forward

def rotatingIntegration(t, a, x0=0., y0=0., vx_0=0., vy_0=0., forward=np.array([1.,0.]), return_velocity=False):
    """Rotate given object space acceleration a and double integrate to world
    space position x.
    
    Returns
    -------
    - x 
    - y
    """
    from vectorRotation import rotatedAcceleration
    t = t.flatten()
    ar = rotatedAcceleration(t, a, vx_0, vy_0, forward)
    #x, y, vx, vy, forward, ar = my_integration(t, a, 0., 0., 1., 0., forward, True)
    xri = integrate.cumtrapz(integrate.cumtrapz(ar[:,0], t, initial=0.)+vx_0, t, initial=0.)+x0
    yri = integrate.cumtrapz(integrate.cumtrapz(ar[:,1], t, initial=0.)+vy_0, t, initial=0.)+y0
# =============================================================================
#     xri = integrate.cumtrapz(integrate.cumtrapz(a[:,0], t, initial=vx_0), t, initial=0.)+x0
#     yri = integrate.cumtrapz(integrate.cumtrapz(a[:,1], t, initial=vy_0), t, initial=0.)+y0
# =============================================================================
    if return_velocity:
        return xri, yri, vx, vy
    else:
        return xri, yri
# =============================================================================
#     #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
#     xri = np.zeros(len(t))
#     yri = np.zeros(len(t))
#     v0 = np.repeat([[vx_0, vy_0]], len(t), axis=0) #rotated starting velocity
#     for i in range(len(t)):
#         #a = rotate(yCircle[i,1:], forward[i,:])
#         forw = np.array([vx[i], vy[i]])
#         rotated_a = rotate(a[i,1:], forw)
#         v0[i,:] = rotate(v0[i,:], forw)
#         #a = rotate(yCircle[i,1:], v[i,:])
#         xri[i] = rotated_a[0]
#         yri[i] = rotated_a[1]
#     xri = integrate.cumtrapz(integrate.cumtrapz(xri, t, initial=0.)+v0[:,0], t, initial=0.)+x0
#     yri = integrate.cumtrapz(integrate.cumtrapz(yri, t, initial=0.)+v0[:,1], t, initial=0.)+y0
#     if return_velocity:
#         return xri, yri, vx, vy
#     else:
#         return xri, yri
# =============================================================================

def accToPos(Xacc, Yacc, x0=0., y0=0., vx_0=0., vy_0=0., forward=np.array([1.0, 0.0])):
    """
    Prameters
    ---------
    x0 is starting position in longitude direction
    y0 is starting position in latitude direction
    """
    return my_integration(Xacc, Yacc, x0, y0, vx_0, vy_0, forward)

def error(x, x0):
    print(x)
    #expand arguments
    #x0, forward = args
    vx_0, vy_0, ax_calibration, ay_calibration, forwardx, forwardy = x
    
    #guess acceleration data at GPS time stamps
    a_calibrated = Yacc + [-9.81, ax_calibration, ay_calibration]
    lon_a, lat_a = accToPos(Xacc, a_calibrated,
                            x0[0], x0[1],
                            vx_0, vy_0,
                            np.array([forwardx, forwardy]))[0:2]
    latlon_a = gpr(Xacc.reshape(-1,1), np.array([lat_a, lon_a]).T, Xgps)[0]
    
    #prepare GPS data
    gps = Data.latlonToMeter(Ygps)
    
    #calculate error (sum of all errors per point has the same order as RMSE)
    residual = (gps - latlon_a)
    error = np.sum(np.sqrt(np.sum(residual * residual, axis=1)))
    print(error)
    return error

def fasterError_old(x, x0, forward):
    """
    x is the tuned parameter and consists of
    
    -   vx_0 (starting velocity in longitudinal direction),
    
    -   vy_0 (starting velocity in latitudinal direction),
    
    -   ax_calibration (sum-calibration of acceleration in longitudinal
        direction),
    
    -   ay_calibration (sum-calibration of acceleration in latitudinal
        direction).
    
    The independent parameters are
    
    -   x0 is a size 2 vector containing the starting position in longitudinal and
        latitudinal direction
    
    -   *forward* is a size 2 vector containing the starting direction in
        longitudinal and latitudinal direction. It is needed for the case of 0
        starting velocity.
    """
    #expand arguments
    #x0, forward = args
    vx_0, vy_0, ax_calibration, ay_calibration = x
    
    #calibrate acc using the tuned parameters
    a_calibrated = Yacc + [-9.81, ax_calibration, ay_calibration]
    
    #integrate using the tuned parameters and given
    lon_a, lat_a = accToPos(Xacc, a_calibrated, x0[0], x0[1], vx_0, vy_0, forward)[0:2]
    
    #guess acceleration data at the last coexisting GPS time stamp
    if Xgps[-1] < Xacc[-1]:
        last = -1
    else:
        last = bisect_left(Xgps, Xacc[-1])
    start = bisect_left(Xacc, Xgps[last])
    if start-5 > 0:
        start = start - 5
    else:
        start = 0
    stop = bisect_right(Xacc, Xgps[last])
    if stop+5 < len(Xacc):
        stop = stop + 5
    else:
        stop = len(Xacc)
    end_a = gpr(Xacc[start:stop].reshape(-1,1),
                np.array([lat_a, lon_a]).T[start:stop,:], [Xgps[last]])[0]
    
    #guess acceleration data at the GPS time stamp in the middle of acc
    middle = bisect_left(Xgps, (Xacc[0] + Xgps[last])/2)
    start = bisect_left(Xacc, Xgps[middle])
    if start-5 > 0:
        start = start - 5
    else:
        start = 0
    stop = bisect_right(Xacc, Xgps[middle])
    if stop+5 < len(Xacc):
        stop = stop + 5
    else:
        stop = len(Xacc)
    mid_a = gpr(Xacc[start:stop].reshape(-1,1),
                np.array([lat_a, lon_a]).T[start:stop,:], [Xgps[middle]])[0]
    
    #stack Acceleration data
    a = np.concatenate((mid_a, end_a))
    #a = end_a
    
    #prepare GPS data
    gps = Data.latlonToMeter(Ygps)[[middle,last],:]
    #gps = Data.latlonToMeter(Ygps)[[last],:]
    
    #calculate error (sum of all errors per point has the same order as RMSE)
    residual = (gps - a)
    error = np.sum(np.sqrt(np.sum(residual * residual, axis=1)))
    print("gps: " + str(gps[1,:]) + " position_acc: " + str(a[1,:]) +\
          " error: " + str(error))
    #print("parameters " + str(x) + " with error "+ str(error))
    return error

def fasterError(x, x0):
    """
    x is the tuned parameter and consists of
    
    -   vx_0 (starting velocity in longitudinal direction),
    
    -   vy_0 (starting velocity in latitudinal direction),
    
    -   ax_calibration (sum-calibration of acceleration in longitudinal
        direction),
    
    -   ay_calibration (sum-calibration of acceleration in latitudinal
        direction).
    
    The independent parameters are
    
    -   x0 is a size 2 vector containing the starting position in longitudinal and
        latitudinal direction
    
    -   *forward* is a size 2 vector containing the starting direction in
        longitudinal and latitudinal direction. It is needed for the case of 0
        starting velocity.
    """
    #expand arguments
    #x0, forward = args
    vx_0, vy_0, ax_calibration, ay_calibration, forwardx, forwardy = x
    
    #calibrate acc using the tuned parameters
    a_calibrated = Yacc + [-9.81, ax_calibration, ay_calibration]
    
    #integrate using the tuned parameters and given
    lon_a, lat_a = accToPos(Xacc, a_calibrated,
                            x0[0], x0[1],
                            vx_0, vy_0,
                            np.array([forwardx, forwardy]))[0:2]
    
    #interpolate acc data to get values at the gps time stamps
    lat_a = np.interp(Xgps, Xacc, lat_a)
    lon_a = np.interp(Xgps, Xacc, lon_a)
    
    #prepare GPS data
    gps = Data.latlonToMeter(Ygps)
    
    #stack Acceleration data
    a = np.c_[lat_a, lon_a]
    
    #calculate error (sum of all errors per point has the same order as RMSE)
    residual = (gps - a)
    error = np.sum(np.sqrt(np.sum(residual * residual, axis=1)))
    print("last gps position: " + str(gps[-1,:]) + " last acc position: " + str(a[-1,:]) +\
          " error: " + str(error))
    #print("parameters " + str(x) + " with error "+ str(error))
    return error# + (100*ax_calibration) ** 2 + (100*ay_calibration) ** 2

#fasterError([1.18053583, 0.73898136, -0.19450797, -0.23301941],[0,0],np.array([1,0]))

def leastSqFit(starting_position=None, fast=True):
    """
    Returns starting parameters for integration as a vector
    [v0_x, v0_y, ax_calibration, ay_calibration].
    
    Use [v0_xm v0_y] as the starting velocity and calibrate Yacc with
    Yacc=Yacc+[ax_calibration, ay_calibration].
    
    You may provide a starting position [lon, lat]. Otherwise it is guessed.
    """
    print("least square fit:")
    print("(With starting position {0})".format(starting_position))
    from SensorFusionGP import initialValues
    gps = Data.latlonToMeter(Ygps)
    v0, forward, moving = initialValues(Xacc, Xgps, gps[:,0], gps[:,1])
    if starting_position is None:
        start = bisect_right(Xgps, Xacc[0])
        if start + 5 < len(Xgps):
            start = start + 5
        else:
            start = len(Xgps)
        latlon_0 = gpr(Xgps[:start,:], gps[:start,:], np.array([Xacc[0:1]]))[0].flatten()
        x0 = latlon_0[::-1]
    else:
        x0 = starting_position
    #v0x, v0y, calibr_ax, calibr_ay, forwardx, forwardy
    initialGuess = np.array([v0[0], v0[1], 0., 0., forward[0], forward[1]])
    from scipy.optimize import minimize
    if fast:
        init = minimize(fasterError, initialGuess, (x0), method='BFGS')
    else:
        init = minimize(error, initialGuess, (x0), method='BFGS')
    print(init)
    print("initial guess was " + str(initialGuess) +\
          " (v0_x, v0_y, calibration_a_x, calibration_a_y)")
    print("used forward vector at start: " + str(forward) +\
          " and starting position: " + str(x0))
    return init.x

def guessInitialParameters(t_a, t_p, p_lat, p_lon):
    """
    get initial values
    (simple wrapper to guard from import loop)
    
    returns v0, forward, moving
    """
    from SensorFusionGP import initialValues
    return initialValues(t_a, t_p, p_lat, p_lon)

def plotIntegration(Yacc, fig=None, ax=None):
    #lat lon zu Meter
    Ygpsnorm = Data.latlonToMeter(Ygps)
    lat, lon = Ygpsnorm[:,0], Ygpsnorm[:,1]
    if fig==None:
        fig=plt.figure()
    if ax==None:
        ax = fig.add_subplot(111, projection='3d')

    
    """plot it"""
    ax.view_init(0, -179)
    ax.plot(Xgps, lon, lat, 'x', label=u'GPS-Position') #plotting ^ lat and > lon
    
    """add gps prediction using 3D-GPR"""
    xGP = np.atleast_2d(np.linspace(min(Xgps), max(Xgps), 1000)).T
    kernel = RBF(10., (0.1, 10)) *\
             ConstantKernel(1.0, (0.01, 10)) *\
             DotProduct(1.,(0.01,100)) *\
             ConstantKernel(1.0, (0.001, 10)) +\
             WhiteKernel(noise_level=4, noise_level_bounds=(0.01, 10))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12, normalize_y=True)
    gp.fit(Xgps, Ygpsnorm)
    print(gp.kernel_)
    samples = gp.sample_y(xGP, 20)
    for i in range(samples.shape[2]):
        latGP = samples[:,0,i]
        lonGP = samples[:,1,i]
        ax.plot(xGP, lonGP, latGP, 'y--', alpha=.5, linewidth=0.5)
    y_pred, sigma = gp.predict(xGP, return_std=True)
    latGP = y_pred[:,0]
    lonGP = y_pred[:,1]
    ax.plot(xGP, lonGP, latGP, 'r-', label=u'Prediction (%s)' % (gp.kernel_))
    
# =============================================================================
#     """get initial values"""
#     #guess forward direction
#     moving = 0
#     for i in range(1, len(x)):
#         dx = lon[i] - lon[i-1]
#         dy = lat[i] - lat[i-1]
#         dt = x[i] - x[i-1]
# #==============================================================================
# #         #we use leastSqFit() now for this
# #         if v0 is None and x[i] > Xacc[0]:
# #             v0 = np.array([dx/dt, dy/dt]).flatten()
# #             print("setting initial speed to"+str(v0))
# #==============================================================================
#         if x[i] > Xacc[0] and isMoving(dx, dy, dt):
#             moving = i
#             #print("calculating initial forward direction...")
#             #print("dx: "+str(dx))
#             #print("dy: "+str(dy))
#             #print("dt: "+str(dt))
#             #print("first noticable movement at iteration: "+str(i))
#             #print("speed in m/s: "+str(sqrt(dx*dx+dy*dy)/dt))
#             forward = np.array([dx, dy])
#             print("calculated initial forward vector: "+str(forward))
#             print("(vector normalization is done during rotation)")
#             break
# =============================================================================
    #guess starting position
    latlon0 = gp.predict(Xacc[0])
    position0 = latlon0[0,::-1] #change to [lon, lat]
    #guess starting velocity and acceleration calibration
    vx_0, vy_0, ax_calibration, ay_calibration, forwardx, forwardy = leastSqFit(position0)
    forward = np.array([forwardx, forwardy])
    calibrate = [0, ax_calibration, ay_calibration]
    print()
    print("calibrating acceleration data by "+str(calibrate))
    Yacc = Yacc + calibrate
    v0 = np.array([vx_0, vy_0]).flatten()
    print("set initial speed to "+str(v0))
    print("\tstarting position to "+str(position0))
    print("\tstarting direction to "+str(forward))
    
    #another 2D plot for the report
    v0_guess, forward_guess, _ = guessInitialParameters(Xacc, Xgps, lat, lon)
    fig_report = plt.figure()
    ax_report = fig_report.add_subplot(111)
    ax_report.set_title("Integrated acceleration")
    x1, y1 = my_integration(Xacc, Yacc-calibrate, position0[0], position0[1], v0_guess[0], v0_guess[1], forward_guess)[0:2]
    x2, y2 = velocity_verlet_integration(Xacc, Yacc-calibrate, position0[0], position0[1], v0_guess[0], v0_guess[1], forward_guess)[0:2]
    x3, y3 = my_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    x4, y4 = velocity_verlet_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax_report.plot(lon, lat, "x", label="GPS positions")
    ax_report.plot(x1, y1, label="recursive approach")
    ax_report.plot(x2, y2, "--", label="simple velocity verlet")
    ax_report.plot(x3, y3, label="recursive approach (fitted)")
    ax_report.plot(x4, y4,  "--", label="simple velocity verlet (fitted)")
    ax_report.set_xlabel("Longitudinal distance from start in $m$")
    ax_report.set_ylabel("Latitudinal distance from start in $m$")
    ax_report.legend(ncol=1)
    fig_report.show()
    
    """double integrate acceleration to position"""
    #lonFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,FORWARD], x=Xacc, initial=0.)+v0[0], x=Xacc, initial=0)
    #latFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,LEFT], x=Xacc, initial=0.)+v0[1], x=Xacc, initial=0)
    #ax.plot(Xacc, lonFromAcc, latFromAcc, 'r-', label=u'$\int\int a_{original}$d$t$ assuming world coordinates (likely wrong)')
    #lonFromAcc, latFromAcc = verlet_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    #ax.plot(Xacc, lonFromAcc, latFromAcc, 'y--', label=u'$\int\int a_{original}$d$t$ (verlet integration)')
    lonFromAcc, latFromAcc = velocity_verlet_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, '--', label=u'$\int\int a_{original}$d$t$ (velocity verlet integration)')
    lonFromAcc, latFromAcc = my_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'b--', label=u'$\int\int a_{original}$d$t$ (my integration)')
    #ax.scatter(x[moving], lon[moving], lat[moving], c="r", label=u'point determining the initial direction')
    
    """GP"""
    t_pred = np.atleast_2d(np.linspace(min(Xacc), max(Xacc), 10000)).T
    #pos_pred, pos_sigma = gpr(Xacc, np.column_stack((x,y)), t_pred)
    acc_pred, acc_sigma = gpr(Xacc.reshape(-1,1), Yacc, t_pred)
    #Normalization here, knowing the car does not move for the first 120
    #data entries is cheating (we better cheat by least square fitting ;-) ):
    #acc_pred[:,1:3] = acc_pred[:,1:3]-np.mean(acc_pred[0:120,1:3], axis=0)
    lonFromAcc, latFromAcc = accToPos(t_pred, acc_pred, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(t_pred, lonFromAcc, latFromAcc, 'g-.', label=u'$\int\int a_{GPR}$d$t$ with object coordinates (my integration)')
    """"""
    ax.set_xlabel(u'time in $s$')
    ax.set_ylabel(u'longitude in $m$')
    ax.set_zlabel(u'latitude in $m$')
    m = max(max(Ygpsnorm.flatten()),
            max(lonFromAcc.flatten()),
            max(latFromAcc.flatten())) * 1.1
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
#==============================================================================
#     ax.set_xlim(0.0,0.1)
#     ax.set_ylim(-0.1, 0.1)
#     ax.set_zlim(-0.1, 0.1)
#==============================================================================
    ax.invert_yaxis()
    #ax.legend()


if __name__ == "__main__":
    
    """test circle"""
    centripetal=-0.2
    N = 0.01
    xCircle = np.array(range(int(100*10**N)))/float(10**N)
    yCircle = np.array([[i, 0.0, centripetal] for i in xCircle])
    fig = plt.figure(figsize=(5,5.3))
# =============================================================================
#     fig.subplots_adjust(bottom=0.15)
#     axis1 = fig.add_axes([0.35, 0.08, 0.50, 0.03], facecolor='lightgoldenrodyellow')
#     axis2 = fig.add_axes([0.35, 0.04, 0.50, 0.03], facecolor='lightgreen', label='resolution')
#     slider1 = Slider(axis1, 'centripetal force', valmin=-1.0, valmax=1.0, valinit=centripetal, valfmt='%0.3f')
#     slider2 = Slider(axis2, 'resolution', valmin=0.01, valmax=2.0, valinit=N, valfmt='%0.5f')
#     slider2.valtext.set_text(round((10**N)*100.0)/100.0)
# =============================================================================
    ax = fig.add_subplot(111)
    ax.set_title("Double integrated object space acceleration")
    #plt.plot(yCircle[:,0], yCircle[:,1])
    xvvi, yvvi = velocity_verlet_integration(xCircle, yCircle, 0., 0., 1., 0.)[0:2]
    line1, = ax.plot(xvvi, yvvi, "g--", label='simple')
    #xvi, yvi = verlet_integration(xCircle, yCircle, 0., 0., 1., 0.)[0:2]
    #line2, = plt.plot(xvi, yvi, "--", label='position with "verlet" integration')
    x, y, vx, vy, forward = my_integration(xCircle, yCircle, 0., 0., 1., 0.)
    line3, = ax.plot(x, y, "b-", label='recursive approach')
    #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
    #xri, yri = rotatingIntegration(xCircle, yCircle, 0., 0., 1., 0.)
    #line4, = plt.plot(xri, yri, "--", label='position with simple double integration of rotated a')
    #test, = plt.plot(vx, vy, label='velocity')
    groundtruth, = ax.plot(np.cos(pi*2*np.array(range(21))/20)/centripetal,
                           (np.sin(pi*2*np.array(range(21))/20)+1)/centripetal,
                           "rx", label='ground truth')
    fig.legend(loc="best", borderaxespad=3)
    fig.tight_layout()
# =============================================================================
#     def update(val):
#         centripetal = slider1.val
#         N = slider2.val
#         xCircle = np.array(range(int(100*10**N)))/float(10**N)
#         slider2.valtext.set_text(round((10**N)*100.0)/100.0)
#         yCircle = np.array([[i, 0.0, centripetal] for i in xCircle])
#         
#         groundtruth.set_xdata(np.cos(pi*2*np.array(range(21))/20)/centripetal)
#         groundtruth.set_ydata((np.sin(pi*2*np.array(range(21))/20)+1)/centripetal)
#         xvvi, yvvi = velocity_verlet_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)[0:2]
#         line1.set_xdata(xvvi)
#         line1.set_ydata(yvvi)
#         #xvi, yvi = verlet_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)[0:2]
#         #line2.set_xdata(xvi)
#         #line2.set_ydata(yvi)
#         x, y, vx, vy, forward = my_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)
#         line3.set_xdata(x)
#         line3.set_ydata(y)
#         #test.set_xdata(vx)
#         #test.set_ydata(vy)
#         #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
#         #xri, yri = rotatingIntegration(xCircle, yCircle, 0., 0., 1., 0.)
#         #line4.set_xdata(xri)
#         #line4.set_ydata(yri)
#         fig.canvas.draw_idle()
#     slider1.on_changed(update)
#     slider2.on_changed(update)
# =============================================================================
    plt.show()
    
    
    #2nd plot: angle over iteration
    plt.figure()
    plt.title("angle of the circular movement (from integrating accel.)")
    plt.plot([180 / pi * atan2(xvvi[i], yvvi[i])
        for i in range(len(forward))], label="velocity verlet integration")
    plt.plot([180 / pi * atan2(vx[i], vy[i])
        for i in range(len(forward))], label="recursive velocity verlet integration")
    plt.ylabel("angle in °")
    plt.legend()
    plt.show()
    
    #3rd plot
    """test integration of real acceleration data vs GPS"""
    fig = plt.figure(figsize=(7.,7.))
    ax = fig.add_subplot(111, projection='3d')
    plotIntegration(Yacc, fig, ax)
    fig.legend(loc='lower left', fontsize='small')
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1, 1, 1]))
    fig.subplots_adjust(top=1.1, right=1.1)
    #fig.tight_layout()
    fig.show()
    

"""    
    #test coordinates
    def plotCoordinateShuffle():
        #
        #+1 +-2
        #   `/
        #+2 +-1
        #   ||
        #-1 +-2
        #   |\
        #-2 +-1
        #   ||
        #
        fig = plt.figure()
        Yacc[:,2] = tmp[:,1]
        Yacc[:,1] = tmp[:,2]
        ax = fig.add_subplot(241, projection='3d')
        ax.set_title("+1+2")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = tmp[:,2]
        Yacc[:,1] = tmp[:,1]
        ax = fig.add_subplot(242, projection='3d')
        ax.set_title("+2+1")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = -tmp[:,1]
        Yacc[:,1] = tmp[:,2]
        ax = fig.add_subplot(243, projection='3d')
        ax.set_title("-1+2")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = -tmp[:,2]
        Yacc[:,1] = tmp[:,1]
        ax = fig.add_subplot(244, projection='3d')
        ax.set_title("-2+1")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = tmp[:,1]
        Yacc[:,1] = -tmp[:,2]
        ax = fig.add_subplot(245, projection='3d')
        ax.set_title("+1-2")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = tmp[:,2]
        Yacc[:,1] = -tmp[:,1]
        ax = fig.add_subplot(246, projection='3d')
        ax.set_title("+2-1")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = -tmp[:,1]
        Yacc[:,1] = -tmp[:,2]
        ax = fig.add_subplot(247, projection='3d')
        ax.set_title("-1-2")
        plotIntegration(Yacc, fig, ax)
        
        Yacc[:,2] = -tmp[:,2]
        Yacc[:,1] = -tmp[:,1]
        ax = fig.add_subplot(248, projection='3d')
        ax.set_title("-2-1")
        plotIntegration(Yacc, fig, ax)
        
        fig.show()
        
    plotCoordinateShuffle()





def accToPos():
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    #vy[0] = 1
    rightHandVector = np.array([1,0])
    for i in range(len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], rightHandVector)
        rightHandVector = Yacc[i, 1:]
        aNext = rotate(Yacc[i+1,1:], rightHandVector)
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[1]*dt*dt
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
    
    plt.figure()
    plt.plot(Yacc[:,0], Yacc[:,1])
    plt.plot(vx, vy)
    plt.plot(x, y)
    plt.show()
"""
