# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:33:13 2018

@author: stephan
"""

import numpy as np
from math import atan2, pi, sqrt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from matplotlib import pyplot as plt
from scipy import integrate


import Data

def gpr(x, y, x_pred):
    mindt = np.min((x[1:]-x[:-1]))
    maxdt = np.max((x[1:]-x[:-1]))
    minY = min(np.min(abs(y), axis=0))
    maxY = max(np.max(abs(y), axis=0))
    avg = (maxY-minY)/(maxdt-mindt)
    kernel = ConstantKernel(avg, (minY/maxdt, maxY/mindt)) *\
             RBF(avg, (minY/maxdt, maxY/mindt))
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(x, y)
    return gp.predict(np.array(x_pred).reshape(-1, 1), return_std=True)

mint = min(min(Data.Xacc), min(Data.Xgps), min(Data.Xgyr))
maxt = max(max(Data.Xacc), max(Data.Xgps), max(Data.Xgyr))
#when to predict
#pred_len = max((len(Data.Xacc), len(Data.Xgps), len(Data.Xgyr)))
#x_pred = np.linspace(mint, maxt, pred_len)
x_pred = Data.Xacc
#predicting
#Yacc = gpr(Data.Xacc, Data.Yacc, x_pred)
Ygyr, Sgyr = gpr(Data.Xgyr, Data.Ygyr, x_pred) #smoothes gyr. data
Xgyr = x_pred
Yacc = Data.Yacc
Xacc = Data.Xacc

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_pred, Ygyr[:,0], "rx", label="x GPR gyr")
ax.plot(x_pred, Ygyr[:,1], "gx", label="y GPR gyr")
ax.plot(x_pred, Ygyr[:,2], "bx", label="z GPR gyr")
ax.plot(Data.Xgyr, Data.Ygyr[:,0], "r-", label="x data")
ax.plot(Data.Xgyr, Data.Ygyr[:,1], "g-", label="y data")
ax.plot(Data.Xgyr, Data.Ygyr[:,2], "b-", label="z data")
ax.fill(np.concatenate([Xgyr, Xgyr[::-1]]), 
        np.concatenate([Ygyr[:,0]-1.9600*Sgyr, (Ygyr[:,0]+1.9600*Sgyr)[::-1]]),
        alpha=.3, ec='None', label="x 95% confidence interval", fc='r')
ax.fill(np.concatenate([Xgyr, Xgyr[::-1]]), 
        np.concatenate([Ygyr[:,1]-1.9600*Sgyr, (Ygyr[:,1]+1.9600*Sgyr)[::-1]]),
        alpha=.3, ec='None', label="y 95% confidence interval", fc='g')
ax.fill(np.concatenate([Xgyr, Xgyr[::-1]]), 
        np.concatenate([Ygyr[:,2]-1.9600*Sgyr, (Ygyr[:,2]+1.9600*Sgyr)[::-1]]),
        alpha=.3, ec='None', label="z 95% confidence interval", fc='b')
ax.legend()
fig.show()

"""
we settled on
x=forward; y=left; z=up
(accordingly roll, pitch, yaw)

but we have
x=down; y=left; z=back
(accordingly yaw, pitch, roll)

so we need to switch them accordingly
"""
FORWARD = 0
ROLL    = 0 #counterclockwise around x-axis
LEFT    = 1
PITCH   = 1 #counterclockwise around y-axis
UP      = 2
YAW     = 2 #counterclockwise around z-axis

acc = np.zeros(Yacc.shape)
acc[:,FORWARD] = -Yacc[:,2]
acc[:,LEFT]    =  Yacc[:,1]
acc[:,UP]      = -Yacc[:,0]
gyr = np.zeros(Ygyr.shape)
gyr[:,ROLL]  = -Ygyr[:,2]
gyr[:,PITCH] =  Ygyr[:,1]
gyr[:,YAW]   = -Ygyr[:,0]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(Xacc, acc[:,FORWARD], label="forward")
ax.plot(Xacc, acc[:,LEFT],    label="left")
ax.plot(Xacc, acc[:,UP],      label="up")
ax.legend()
ax = fig.add_subplot(212)
ax.plot(Xgyr, gyr[:,ROLL],  label="roll")
ax.plot(Xgyr, gyr[:,PITCH], label="pitch")
ax.plot(Xgyr, gyr[:,YAW],   label="yaw")
ax.legend()
fig.show()

#standard complementary filter to eliminate drift of gyr. integration
def filterStepSync(roll, pitch, acc, gyr, dt):
    roll  = roll  + gyr[ROLL]  * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
    pitch = pitch + gyr[PITCH] * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
    
    #since angle calculation from accelerometer depends heavily on g dominating
    #make sure there is no greater force in another direction
    sumacc = sum(np.abs(acc))
    if sumacc > 5 and sumacc < 15:
        #when acc[UP] << 0 and acc[LEFT] ~ 0 then rollAcc = 0
        #when acc[LEFT] > 0 then rollAcc > 0 (counterclockwise around forward vector)
        rollAcc = atan2(acc[LEFT], -acc[UP]) * 180.0 / pi  #TODO +/-acc[LEFT] ?
        #when acc[UP] << 0 and acc[FORWARD] ~ 0 then rollACC = 0
        #when acc[FORWARD] > 0 then rollAcc < 0 (counterclockwise around left vector)
        pitchAcc = atan2(-acc[FORWARD], -acc[UP]) * 180.0 / pi
        
        roll  = 0.99*roll  + 0.01*rollAcc
        pitch = 0.99*pitch + 0.01*pitchAcc
    
    return roll, pitch

#my try for async complementary filter
def filterStepAsync(roll, pitch, acc=None, gyr=None, dt=None):
    """
    aufteilen in 2 Aufrufe
    angle += gyr*dt
    """
    if gyr is not None:
        roll  = roll  + gyr[ROLL]  * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
        pitch = pitch + gyr[PITCH] * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
    
    """
    und
    roll = 0.98*roll + 0.02*atan(acc_x/acc_z)*180/pi
    Formel (3.14) und (3.15) aus Feng, Shimin (2014)
    "Sensor fusion with Gaussian processes"
    """
    if acc is not None:
        #since angle calculation from accelerometer depends heavily on g dominating
        #make sure there is no greater force in another direction
        sumacc = sum(np.abs(acc))
        if sumacc > 5 and sumacc < 15:
            #rollAcc = atan2(acc[LEFT], -acc[UP]) * 180.0 / pi
            rollAcc = atan2(
                            acc[LEFT],
                            np.sign(-acc[UP]) * sqrt(0.01*acc[FORWARD]**2 + acc[UP]**2)
                      ) * 180.0 / pi
            #pitchAcc = atan2(-acc[FORWARD], -acc[UP]) * 180.0 / pi
            pitchAcc = atan2(-acc[FORWARD], sqrt(acc[LEFT]**2 + acc[UP]**2)) * 180.0 / pi
            roll  = 0.99*roll  + 0.01*rollAcc
            pitch = 0.99*pitch + 0.01*pitchAcc
    return roll, pitch

roll = np.zeros(len(acc))
pitch = np.zeros(len(acc))
rollSimple = np.zeros(len(acc))
pitchSimple = np.zeros(len(acc))
for i in range(len(Xgyr)-1):
    #Zeit, die beschleunigt wird
    dt = Xgyr[i+1] - Xgyr[i]
    roll[i+1], pitch[i+1] = \
        filterStepAsync(roll[i], pitch[i], acc[i,:], gyr[i,:], dt)
    rollSimple[i+1], pitchSimple[i+1] = \
        filterStepSync(rollSimple[i], pitchSimple[i], acc[i,:], gyr[i,:], dt)
rollIntGyr  = integrate.cumtrapz(gyr[:,ROLL],  x=Xgyr[:,0], initial=0.0) * 180.0 / pi
pitchIntGyr = integrate.cumtrapz(gyr[:,PITCH], x=Xgyr[:,0], initial=0.0) * 180.0 / pi
rollAcc = [(atan2(acc[i,LEFT],
                   np.sign(-acc[i,UP]) * sqrt(0.01*acc[i,FORWARD]**2 + acc[i,UP]**2)
                ) * 180.0 / pi)
                for i in range(len(Xacc))]
pitchAcc  = [(atan2(-acc[i,FORWARD],
                   sqrt(acc[i,LEFT]**2 + acc[i,UP]**2)
                ) * 180.0 / pi)
                for i in range(len(Xacc))]
rollAccSimple = [(atan2(acc[i,LEFT], -acc[i,UP]) * 180.0 / pi)
                for i in range(len(Xacc))]
pitchAccSimple  = [(atan2(-acc[i,FORWARD], -acc[i,UP]) * 180.0 / pi)
                for i in range(len(Xacc))]
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_title("Pitch")
ax.plot(Xgyr, pitch,             label="Complementary Filter [Feng 2014]")
ax.plot(Xgyr, pitchSimple, "--", label="Complementary Filter (simple)")
ax.plot(Xgyr, pitchIntGyr,       label="Gyroscope integr.")
ax.plot(Xacc, pitchAcc,          label="Accelerometer [Feng 2014]")
ax.plot(Xacc, pitchAccSimple,    label="Accelerometer (simple)")
ax.legend()
ax = fig.add_subplot(122)
ax.set_title("Roll")
ax.plot(Xgyr, roll,             label="Complementary Filter [Feng 2014]")
ax.plot(Xgyr, rollSimple, "--", label="Complementary Filter (simple)")
ax.plot(Xgyr, rollIntGyr,       label="Gyroscope integr.")
ax.plot(Xacc, rollAcc,          label="Accelerometer [Feng 2014]")
ax.plot(Xacc, rollAccSimple,    label="Accelerometer (simple)")
ax.legend()
fig.show()




