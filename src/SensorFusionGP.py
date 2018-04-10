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
from myGP import kernel
from vectorRotation import rotatedAcceleration
from Integration import isMoving

n = 1000

#for 1 dimension at a time!
def derivative_GP(X_pos, X_acc, Y_pos, Y_acc, s1, l, s2, noiseGPS = 4., noiseAcc = 0.2):
    #X_pred = [X_pos X_acc]
    X_pred = np.concatenate((X_pos, X_acc))
    sort = X_pred[:,0].argsort()
    unique = np.ones(len(X_pred), dtype=bool)
    unique2 = np.ones(len(X_pred), dtype=bool)
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique[1:])
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique2[:-1])
    dt = X_pred[sort][unique][1:]-X_pred[sort][unique][:-1]
    
    #test = X_pred[sort][unique]
    #training = np.concatenate((X_pos, X_acc))
    test = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), n)).T
    training = X_pred[sort][unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    median_dt = np.median(dt)
    training = np.r_[training, [training[-1]+median_dt], [training[-1]+median_dt*2.0]]
    
    #Set the kernel which chooses the observed values from latent variables.
    #choose Kp such that Y_pos = Kp @ latentY(X_pred).
    Kp = np.zeros((len(X_pos), len(X_pred)))
    for i in range(Kp.shape[1]):
        if sort[i] < len(X_pos):
            Kp[sort[i], i] = 1.
    Kp = Kp[:,unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    Kp = np.c_[Kp, np.zeros((len(X_pos),2))]
    
    #choose Ka such that Y_acc = Ka @ latentY(X_pred)
    Ka = np.zeros((len(X_acc), len(training)))
    for i in range(Ka.shape[1]-2):
        j = sort[unique2][i] - len(X_pos)
        if j >= 0:
            dt = training[i+1] - training[i]
            Ka[j, i] = 1./(dt*dt)
            dt2 = training[i+2] - training[i+1]
            Ka[j, i+1] = -1./(dt*dt)-1./(dt*dt2)
            Ka[j, i+2] = 1./(dt*dt2)
    
    #M = [Y_pos Y_acc]
    M = np.concatenate((Y_pos, Y_acc))
    #K = [Kp Ka]
    K = np.concatenate((Kp, Ka))
    
    #now we have set M = K @ latentY and can try to find this latent Y
    S = kernel(training, training, s1, l, s2)
    S2 = kernel(test, test, s1, l)
    S12 = kernel(test, training, s1, l, s2)
    S21 = kernel(training, test, s1, l, s2)
    noise = [noiseGPS*noiseGPS]*len(Y_pos)+[noiseAcc*noiseAcc]*len(Y_acc)
    mu2_1 = S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    S2_1 = S2 - S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ K @ S21
    #resuts from setting K2 to I [Shimin Feng 2014]
    #mu2_1 = I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    #S2_1 = S2 - I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T) @ K @ S21 @ I.T
    #in the original equations [Murray-Smith 2005]
    #mu2_1 = K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ M1
    #S2_1 = S2 - K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ K1 @ S21 @ K2
    return test, mu2_1, S2_1

def initialValues(t_a, t_p, p_lat, p_lon):
    """get initial values"""
    v0 = None
    moving = 0
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
        a[1,:] = curve
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
        j = bisect_left(t, t_a[1])
        dt = np.linspace(0., 1., 20)
        dp = np.cumsum(np.cumsum([a[1]/20]*20))
        ax.plot((t[j] + dt), (mu[j] + dp), "r-", label='curvature')
        for i in range(int(len(a)/10),len(a), int(len(a)/10)):
            j = bisect_left(t, t_a[i])
            dt = np.linspace(0., 1., 20)
            dp = np.cumsum(np.cumsum([a[i]/20]*20))
            ax.plot((t[j] + dt), (mu[j] + dp), "r-")
        ax.fill(np.concatenate([t, t[::-1]]),
                         np.concatenate([mu.flatten() - 1.9600 * np.diag(sigma),
                                        (mu.flatten() + 1.9600 * np.diag(sigma))[::-1]]),
                         alpha=.3, fc='b', ec='None', label='95% confidence interval')
        ax.fill(np.concatenate([t, t[::-1]]),
                         np.concatenate([mu.flatten() - np.diag(sigma),
                                        (mu.flatten() + np.diag(sigma))[::-1]]),
                         alpha=.3, fc='g', ec='None', label='standard deviation')
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
        s2 = 1.
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
        ssigma  = Slider(axsigma,  'sigma', valmin=0.01, valmax=20., valinit=s1, valfmt='%0.2f')
        ssigma2 = Slider(axsigma2, 'sigma2', valmin=0.01, valmax=1., valinit=s2, valfmt='%0.2f')
        slength = Slider(axlength, 'length scale', valmin=0.01, valmax=10., valinit=l, valfmt='%0.4f')
        snoiseP = Slider(axnoiseP, 'noise of data', valmin=1e-7, valmax=10., valinit=noiseP, valfmt='%0.5f')
        snoiseA = Slider(axnoiseA, 'noise of curvature', valmin=1e-7, valmax=100., valinit=noiseA, valfmt='%0.5f')
        
        TEST(t_p, p, t_a, a, curve, s1, l, s2, noiseP, noiseA, ax)
        ax.set_ylim(np.min(p)-100, np.max(p)+100)
        def updateTEST(val):
            ax.clear()
            TEST(t_p, p, t_a, a, scurve.val, ssigma.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val, ax)
            ax.set_ylim(np.min(p)-100, np.max(p)+100)
            fig.canvas.draw_idle()
        
        scurve.on_changed(updateTEST)
        ssigma.on_changed(updateTEST)
        ssigma2.on_changed(updateTEST)
        slength.on_changed(updateTEST)
        snoiseP.on_changed(updateTEST)
        snoiseA.on_changed(updateTEST)
        return [scurve, ssigma, ssigma2, slength, snoiseP, snoiseA]
        
    #TEST_plot_with_sliders()
    
    gps = Data.latlonToMeter(Data.Ygps)
    v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, gps[:,0], gps[:,1])
    acc = rotatedAcceleration(Data.Xacc, Data.Yacc, v0[0], v0[1], forward)
    keeping_sliders_responsive = TEST_plot_with_sliders(
                            Data.Xgps.reshape(-1,1),
                            gps[:,1].reshape(-1,1),
                            Data.Xacc.reshape(-1,1),
                            acc[:,1].reshape(-1,1))
    keeping_sliders_responsive2 = TEST_plot_with_sliders(
                            Data.Xgps.reshape(-1,1),
                            gps[:,0].reshape(-1,1),
                            Data.Xacc.reshape(-1,1),
                            acc[:,0].reshape(-1,1))