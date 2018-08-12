# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:33:13 2018

@author: stephan
"""

import numpy as np
from math import atan2, pi, sqrt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct, ExpSineSquared
from matplotlib import pyplot as plt
from scipy import integrate


import Data

def gpr(x, y, x_pred, return_std=True, return_kernel=False, exp_sine_sq_kernel=False):
    mindt = np.min((x[1:]-x[:-1]))
    maxdt = np.max((x[1:]-x[:-1]))
    minY = min(np.min(y, axis=0))
    maxY = max(np.max(y, axis=0))
    avgY = np.average(np.average(y, axis=0))
    if exp_sine_sq_kernel:
        kernel = ConstantKernel(avgY**2, (minY**2, maxY**2)) *\
             ExpSineSquared(1, 2.5*(maxdt-mindt),
                            (1.0*mindt, 10.0*maxdt),
                            periodicity_bounds=(1e-6, 1e+2)) +\
             WhiteKernel()
    else:
        kernel = ConstantKernel(avgY**2, (minY**2, maxY**2)) *\
             RBF(2.5*(maxdt-mindt), (1.0*mindt, 10.0*maxdt)) +\
             WhiteKernel()
    print("kernel: ", end='')
    print(kernel)
    print(kernel.get_params())
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(x, y)
    if return_std:
        x_predicetd, s_predicted = gp.predict(np.array(x_pred).reshape(-1, 1), return_std=return_std)
        if return_kernel:
            return x_predicetd, s_predicted, gp.kernel_
        else:
            return x_predicetd, s_predicted
    else:
        x_predicetd = gp.predict(np.array(x_pred).reshape(-1, 1), return_std=return_std)
        if return_kernel:
            return x_predicetd, gp.kernel_
        else:
            return x_predicetd

mint = min(min(Data.Xacc), min(Data.Xgps), min(Data.Xgyr))
maxt = max(max(Data.Xacc), max(Data.Xgps), max(Data.Xgyr))
#when to predict
#pred_len = max((len(Data.Xacc), len(Data.Xgps), len(Data.Xgyr)))
#x_pred = np.linspace(mint, maxt, pred_len)
x_pred = Data.Xacc_corrected2
#predicting
#Yacc = gpr(Data.Xacc, Data.Yacc, x_pred)
Xgyr = x_pred
Yacc = Data.Yacc
Xacc = Data.Xacc_corrected2

#---plot 1 (correlation plot matrix)
import pandas as pd
from pandas.plotting import scatter_matrix
Ygyr, Sgyr, kernel = gpr(Data.Xgyr_corrected2, Data.Ygyr, Data.Xacc_corrected2, True, True)
print("Fitted parameters: ", end='')
print(kernel)
dat = np.concatenate((Data.Yacc, Ygyr), axis=1)
dat_deriv = np.concatenate((dat[0:1,:], dat[1:,:]-dat[:-1,:]))
df = pd.DataFrame(np.concatenate((dat,dat_deriv), axis=1),
                  columns=("UP","FORWARD","LEFT",
                           "YAW","ROLL","PITCH",
                           "UP/dt","FORWARD/dt","LEFT/dt",
                           "YAW/dt","ROLL/dt","PITCH/dt"))
scatter_matrix(df)


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
acc[:,FORWARD] =  Yacc[:,Data.FORWARD]
acc[:,LEFT]    =  Yacc[:,Data.LEFT]
acc[:,UP]      =  Yacc[:,Data.UP]
gyr = np.zeros(Ygyr.shape)
gyr[:,ROLL]  =  Ygyr[:,Data.ROLL]
gyr[:,PITCH] =  -Ygyr[:,Data.PITCH]
gyr[:,YAW]   =  Ygyr[:,Data.YAW]


#---plot 2
fig = plt.figure()
ax = fig.add_subplot(411)
ax.plot(Xacc, acc[:,UP], "r", label="up")
ax.set_ylabel("a in $m/s$", color="r")
ax2 = ax.twinx()
ax2.plot(Xgyr, gyr[:,PITCH], "b", label="pitch")
ax2.set_ylabel(u"$\omega$ in °", color="b")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax = fig.add_subplot(412)
ax.plot(Xacc, acc[:,LEFT], "r", label="left")
ax.set_ylabel("a in $m/s$", color="r")
ax2 = ax.twinx()
ax2.plot(Xgyr, gyr[:,YAW], "b", label="yaw")
ax2.set_ylabel(u"$\omega$ in °", color="b")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax = fig.add_subplot(413)
ax.plot(Xacc, acc[:,LEFT], "r", label="left")
ax.set_ylabel("a in $m/s$", color="r")
ax2 = ax.twinx()
ax2.plot(Xgyr, gyr[:,ROLL], "b", label="roll")
ax2.set_ylabel(u"$\omega$ in °", color="b")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
fig.show()

ax = fig.add_subplot(414)
ax.plot(Xacc, acc[:,FORWARD], "r", label="forward")
ax.set_ylabel("a in $m/s$", color="r")
ax2 = ax.twinx()
ax2.plot(Xgyr, gyr[:,PITCH], "b", label="pitch")
ax2.set_ylabel(u"$\omega$ in °", color="b")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

#standard complementary filter to eliminate drift of gyr. integration
def filterStepSync(roll, pitch, acc, gyr, dt):
    roll  = roll  + gyr[ROLL]  * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
    pitch = pitch + gyr[PITCH] * dt * 180.0 / pi #TODO +/- ? (gyr countercl. or clockw.)
    
    #since angle calculation from accelerometer depends heavily on g dominating
    #make sure there is no greater force in another direction
    sumacc = sum(acc**2)**.5
    if True:#sumacc > 5 and sumacc < 15:
        #when acc[UP] << 0 and acc[LEFT] ~ 0 then rollAcc = 0
        #when acc[LEFT] > 0 then rollAcc > 0 (counterclockwise around forward vector)
        rollAcc = atan2(acc[LEFT], acc[UP]) * 180.0 / pi  #TODO +/-acc[LEFT] ?
        #when acc[UP] << 0 and acc[FORWARD] ~ 0 then rollACC = 0
        #when acc[FORWARD] > 0 then rollAcc < 0 (counterclockwise around left vector)
        pitchAcc = atan2(-acc[FORWARD], acc[UP]) * 180.0 / pi
        
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
        sumacc = sum(acc**2)**.5
        if True:#sumacc > 5 and sumacc < 9.5:#15:
            #rollAcc = atan2(acc[LEFT], -acc[UP]) * 180.0 / pi
            rollAcc = atan2(
                            acc[LEFT],
                            np.sign(acc[UP]) * sqrt(0.01*acc[FORWARD]**2 + acc[UP]**2)
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
                   np.sign(acc[i,UP]) * sqrt(0.01*acc[i,FORWARD]**2 + acc[i,UP]**2)
                ) * 180.0 / pi)
                for i in range(len(Xacc))]
pitchAcc  = [(atan2(-acc[i,FORWARD],
                   sqrt(acc[i,LEFT]**2 + acc[i,UP]**2)
                ) * 180.0 / pi)
                for i in range(len(Xacc))]
rollAccSimple = [(atan2(acc[i,LEFT], acc[i,UP]) * 180.0 / pi)
                for i in range(len(Xacc))]
pitchAccSimple  = [(atan2(-acc[i,FORWARD], acc[i,UP]) * 180.0 / pi)
                for i in range(len(Xacc))]
#my roll estimate
testRoll = np.zeros(len(Xacc))
for i in range(len(Xgyr)-1):
    testRoll[i+1] = 0.97*(testRoll[i] + 180./pi*gyr[i,ROLL]*(Xgyr[i+1]-Xgyr[i]))

#---plot 4
fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Pitch")
ax.plot(Xgyr, pitch,              label="Complementary Filter [Feng 2014]")
ax.plot(Xgyr, pitchSimple, "--",  label="Complementary Filter (simple)")
ax.plot(Xgyr, pitchIntGyr,        label="Gyroscope integr.")
ax.plot(Xacc, pitchAcc,           label="Accelerometer [Feng 2014]", linewidth=0.75)
ax.plot(Xacc, pitchAccSimple,"--",label="Accelerometer (simple)", linewidth=0.75)
ax.set_ylabel(u"angle in $°$")
ax = fig.add_subplot(212)
ax.set_title("Roll")
ax.plot(Xgyr, roll,              label="Complementary Filter [Feng 2014]")
ax.plot(Xgyr, rollSimple, "--",  label="Complementary Filter (simple)")
ax.plot(Xgyr, rollIntGyr,        label="Gyroscope integr.")
ax.plot(Xacc, rollAcc,           label="Accelerometer [Feng 2014]", linewidth=0.75)
ax.plot(Xacc, rollAccSimple,"--",label="Accelerometer (simple)", linewidth=0.75)
#ax.plot(Xacc, testRoll,label="Accelerometer TEST")
#ax.plot(Xacc, (np.arctan2(9.5-acc[:,UP]-acc[:,FORWARD],acc[:,UP])*180/pi) ,label="Accelerometer TEST2")
ax.set_xlabel("time in $s$")
ax.set_ylabel(u"angle in $°$")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2)
fig.tight_layout()
fig.show()


