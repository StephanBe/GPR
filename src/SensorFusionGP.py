# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:43:49 2018

@author: Stephan
"""

import numpy as np
from matplotlib import pyplot
from matplotlib.widgets import Slider
from bisect import bisect_left

import Data
from myGP import derivative_GP
from vectorRotation import rotatedAcceleration
from Integration import isMoving

n = 1000

def initialValues(t_a, t_p, p_lat, p_lon):
    """
    get initial values
    
    returns v0, forward, moving
    """
    v0 = None
    moving = 0
    forward = np.array([float('nan'), float('nan')])
    for i in range(1, len(t_p)):
        dx = p_lon[i] - p_lon[i-1]
        dy = p_lat[i] - p_lat[i-1]
        dt = t_p[i] - t_p[i-1]
        if t_p[i] > t_a[0]:
            if v0 is None:
                v0 = np.array([dx/dt, dy/dt]).flatten()
            if isMoving(dx, dy, dt):
                moving = i
                forward = np.array([dx, dy])
                break
    return v0, forward, moving

if __name__ == "__main__":
    def TEST(t_p, p, t_a, a, curve, s1 = 2., l = 1.0, s2 = 1.0, noiseP = 0.1, noiseA = 1., ax=None):        
        from TestingIntegrationMatrices import derivative_GP as derivative_GP_test
        curve_index = len(t_a)>>1
        a[curve_index,:] = curve
        if ax is None:
            ax = pyplot.figure().add_subplot(111)  
        t, mu, sigma = derivative_GP(X_pos=t_p, Y_pos=p, X_acc=t_a, Y_acc=a,
                               s1 = s1,
                               l  = l,
                               s2 = s2,
                               noiseGPS = noiseP,
                               noiseAcc = noiseA)
        
        ax.plot(t, mu, label="prediction")
        ax.plot(t_p, p, label="data")
        for i in range(int(len(a)/10),len(a), int(np.ceil(len(a)/10))):
            j = bisect_left(t, t_a[i])
            dt = np.linspace(0., 1., 20)
            dp = np.cumsum(np.cumsum([a[i]/20]*20))
            ax.plot((t[j] + dt), (mu[j] + dp), "y-")
        j = bisect_left(t, t_a[curve_index])
        dt = np.linspace(0., 1., 20)
        dp = np.cumsum(np.cumsum([a[curve_index]/20]*20))
        
        ax.plot((t[j] + dt), (mu[j] + dp), "r-", label='curvature')
        ax.fill(np.concatenate([t, t[::-1]]),
                         np.concatenate([mu.flatten() - 1.9600 * np.diag(sigma),
                                        (mu.flatten() + 1.9600 * np.diag(sigma))[::-1]]),
                         alpha=.3, fc='b', ec='None', label='95% confidence interval')
        ax.fill(np.concatenate([t, t[::-1]]),
                         np.concatenate([mu.flatten() - np.diag(sigma),
                                        (mu.flatten() + np.diag(sigma))[::-1]]),
                         alpha=.3, fc='g', ec='None', label='standard deviation')
        
        t, mu, sigma = derivative_GP_test(X_pos=t_p, Y_pos=p, X_acc=t_a, Y_acc=a,
                               s1 = s1,
                               l  = l,
                               s2 = s2,
                               noisePos = noiseP,
                               noiseAcc = noiseA)
        ax.plot(t, mu, label="prediction (test)")
        
        ax.legend()
        return ax
    
    def TEST_plot_with_sliders(t_p=np.array([[.0],[.1],[1.5],[2.]]),
             p = np.array([[.0],[.5],[-1.5],[0.5]]),
             t_a = np.array([[0.],[1.1],[2.]]),
             a = np.array([[-1.],[-1.5],[-2.]]),
             curve=-1.5
        ):
        curve = a[1,:]
        s1 = 10.
        s2 = 0.
        l = 0.5
        noiseP = 4.
        noiseA = 50.
        fig = pyplot.figure()
        fig.subplots_adjust(bottom=0.3)
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        axcurve  = fig.add_axes([0.38,0.21,0.50,0.03])
        axsigma  = fig.add_axes([0.38,0.17,0.50,0.03])
        axsigma2 = fig.add_axes([0.38,0.13,0.50,0.03])
        axlength = fig.add_axes([0.38,0.09,0.50,0.03])
        axnoiseP = fig.add_axes([0.38,0.05,0.50,0.03])
        axnoiseA = fig.add_axes([0.38,0.01,0.50,0.03])
        scurve  = Slider(axcurve,  'curve', valmin=-100, valmax=100., valinit=curve, valfmt='%0.2f')
        ssigma  = Slider(axsigma,  'sigma', valmin=0.01, valmax=200., valinit=s1, valfmt='%0.2f')
        ssigma2 = Slider(axsigma2, 'sigma2', valmin=0., valmax=200., valinit=s2, valfmt='%0.2f')
        slength = Slider(axlength, 'length scale', valmin=0.01, valmax=20., valinit=l, valfmt='%0.4f')
        snoiseP = Slider(axnoiseP, 'noise of data', valmin=1e-7, valmax=1000., valinit=noiseP, valfmt='%0.5f')
        snoiseA = Slider(axnoiseA, 'noise of curvature', valmin=0, valmax=100., valinit=noiseA, valfmt='%0.5f')
        
        TEST(t_p, p, t_a, a, curve, s1, l, s2, noiseP, noiseA, ax)
        ax.set_ylim(np.min(p)-100, np.max(p)+100)
        def updateTEST(val):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.clear()
            TEST(t_p, p, t_a, a, scurve.val, ssigma.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val, ax)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            #ax.set_ylim(np.min(p)-100, np.max(p)+100)
            fig.canvas.draw_idle()
        
        scurve.on_changed(updateTEST)
        ssigma.on_changed(updateTEST)
        ssigma2.on_changed(updateTEST)
        slength.on_changed(updateTEST)
        snoiseP.on_changed(updateTEST)
        snoiseA.on_changed(updateTEST)
        return [scurve, ssigma, ssigma2, slength, snoiseP, snoiseA]
    
    #plot 1
    keeping_sliders_responsive0 = TEST_plot_with_sliders()
    
    gps = Data.latlonToMeter(Data.Ygps)
    v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, gps[:,0], gps[:,1])
    
    acc_obj = np.copy(Data.Yacc)
# =============================================================================
#     #vorher: android x (Yacc[:,0] = nach unten beschleunigen)
#     #vorher: android y (Yacc[:,1] = nach links beschleunigen)
#     #vorher: android z (Yacc[:,2] = nach hinten beschleunigen)
#     DOWN = 0
#     FORWARD = 1
#     LEFT = 2
#     acc_obj[:,DOWN] = Data.Yacc[:,0]
#     acc_obj[:,FORWARD] = -Data.Yacc[:,2]
#     acc_obj[:,LEFT] = Data.Yacc[:,1]
# =============================================================================
    
    acc = rotatedAcceleration(Data.Xacc, acc_obj, v0[0], v0[1], forward)
    
    #plot 2
    keeping_sliders_responsive1 = TEST_plot_with_sliders(
                            Data.Xgps.reshape(-1,1),
                            gps[:,1].reshape(-1,1),
                            Data.Xacc_corrected2.reshape(-1,1),
                            acc[:,0].reshape(-1,1))
    
    #plot 3
    keeping_sliders_responsive2 = TEST_plot_with_sliders(
                            Data.Xgps.reshape(-1,1),
                            gps[:,0].reshape(-1,1),
                            Data.Xacc_corrected2.reshape(-1,1),
                            acc[:,1].reshape(-1,1))
    
    #plot 4 position from integration (with simple initial values)
    from Integration import rotatingIntegration
    xri, yri = rotatingIntegration(Data.Xacc, acc_obj, 0., 0., v0[0], v0[1], forward)
    pyplot.figure()
    pyplot.plot(xri, yri, label="position from integration")
    pyplot.plot(gps[:,1], gps[:,0], label="position from gps")
    pyplot.plot([gps[moving,1], gps[moving,1]+forward[0]],
                [gps[moving,0], gps[moving,0]+forward[1]],
                "->", label="initial direction")
    pyplot.legend()
    
    
    #plot 5
    from Integration import my_integration
    v = my_integration(Data.Xacc, Data.Yacc, 0, 0, v0[0], v0[1], forward)[4]
    pyplot.figure()
    pyplot.title("Comparing the velocities calculated from GPS and Accelerometer data")
    pyplot.plot(Data.Xacc, v[:,0], "r-", label="$v_x$ from acc")
    pyplot.plot(Data.Xacc, v[:,1], "b-", label="$v_y$ from acc")
    v = (gps[1:,:]-gps[:-1,:])/(Data.Xgps[1:,:]-Data.Xgps[:-1,:])
    v = np.insert(v, 0, v0[[1,0]], axis=0)
    #a = (v[1:,:]-v[:-1,:])/(Data.Xgps[1:,:]-Data.Xgps[:-1,:])
    #a = np.insert(a, 0, [0, 0], axis=0)
    pyplot.plot(Data.Xgps, v[:,1], "r--", label="$v_{lon}$ from gps")
    pyplot.plot(Data.Xgps, v[:,0], "b--", label="$v_{lat}$ from gps")
    pyplot.xlabel("$t$ in $s$")
    pyplot.ylabel("$v$ in $m/s$")
    pyplot.legend()
    
    #plot 6 fusing integrated velocity with GPS position
    from myGP import asynchronous_GP
    s = 10.
    l = 1.
    s2 = 1.
    noiseGPS = 2.
    noiseAcc = 20.
    x, meanLon, errLon = asynchronous_GP(Data.Xgps, Data.Xacc, gps[:,1], xri, s, l, s2, noiseGPS, noiseAcc)
    x, meanLat, errLat = asynchronous_GP(Data.Xgps, Data.Xacc, gps[:,0], yri, s, l, s2, noiseGPS, noiseAcc)
    pyplot.figure()
    pyplot.title("Fusing integrated velocity with GPS position into a single heteroscedastic GP")
    pyplot.plot(gps[:,1], gps[:,0], label="GPS position")
    pyplot.plot(xri,      yri,      label="position from Accelerometer")
    pyplot.plot(meanLon, meanLat, label="predicted")
    pyplot.fill(np.concatenate([meanLon, meanLon[::-1]]),
                 np.concatenate([meanLat.flatten() - 1.9600 * np.diag(errLon),
                                (meanLat.flatten() + 1.9600 * np.diag(errLat))[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
    pyplot.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    