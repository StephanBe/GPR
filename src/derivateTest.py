# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 00:08:10 2018

@author: stephan
"""

from matplotlib import pyplot
import numpy as np
import Data
import Integration
from vectorRotation import rotate

#Squared Exponential Kernel

def sqared_exponential_kernel(a, b, s=1, l=1):
    """
    Squared Exponential Kernel.
    """
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2,1) - 2*(a @ b.T)
    return s**2 * np.exp(-1/(2 * l**2) * sqdist)

#Linear Kernel

def linear_kernel(x1, x2, s1=0, s2=1, c=0):
    """
    Squared Exponential Kernel.
    
    c is the starting point for the linear prior.
    """
    return s1**2 + (s2**2) * (x1-c)*(x2-c).T

def kernel(a, b, s=1, l=1):
    """
    Currently used all purpose kernel.
    """
    return sqared_exponential_kernel(a, b, s, l)# + linear_kernel(a, b)

#for 1 dimension at a time!
def derivative_GP(X_pos, X_acc, Y_pos, Y_acc):
    #X_pred = [X_pos X_acc]
    X_pred = np.concatenate((X_pos, X_acc))
    sort = X_pred[:,0].argsort()
    unique = np.ones(len(X_pred), dtype=bool)
    unique2 = np.ones(len(X_pred), dtype=bool)
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique[1:])
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique2[:-1])
    #Set the kernel which chooses the observed values from latent variables.
    #choose Kp such that Y_pos = Kp @ latentY(X_pred)
    Kp = np.zeros((len(X_pos), len(X_pred) plus zwei))
    for i in range(len(X_pred)):
        if sort[i] < len(X_pos):
            Kp[sort[i], i] = 1.
    Kp = Kp[:,unique]
    #choose Ka such that Y_acc = Ka @ latentY(X_pred)
    dt = X_pred[sort][unique][1:]-X_pred[sort][unique][:-1]
    Ka = np.zeros((len(X_acc), len(X_pred[unique2])))
    for i in range(len(X_pred[unique2])):
        j = sort[unique2][i] - len(X_pos)
        if j >= 0:
            if i < (len(X_pred[unique2])-2):
                dt = X_pred[sort][unique][i+1] - X_pred[sort][unique][i]
                Ka[j, i] = 1./(dt*dt)
                dt2 = X_pred[sort][unique][i+2] - X_pred[sort][unique][i+1]
                Ka[j, i+1] = -1./(dt*dt)-1./(dt*dt2)
                Ka[j, i+2] = 1./(dt*dt2)
    #M = [Y_pos Y_acc]
    M = np.concatenate((Y_pos, Y_acc))
    #K = [Kp Ka]
    K = np.concatenate((Kp, Ka))
    #now we have set M = K @ latentY and can try to find this latent Y
    #test = X_pred[sort][unique]
    #training = np.concatenate((X_pos, X_acc))
    test = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), 1000)).T
    training = X_pred[sort][unique]
    S = kernel(training, training)
    S2 = kernel(test, test)
    S12 = kernel(test, training)
    S21 = kernel(training, test)
    noiseGPS = [4]*len(Y_pos)
    noiseAcc = [0.1]*len(Y_acc)
    noise = noiseGPS+noiseAcc
    mu2_1 = S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    S2_1 = S2 - S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ K @ S21
    #resuts from setting K2 to I [Shimin Feng 2014]
    #mu2_1 = I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    #S2_1 = S2 - I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T) @ K @ S21 @ I.T
    #in the original equations [Murray-Smith 2005]
    #mu2_1 = K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ M1
    #S2_1 = S2 - K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ K1 @ S21 @ K2
    return test, mu2_1, S2_1

if __name__ == "__main__":
    FORWARD = Integration.FORWARD
    LEFT = Integration.LEFT
    #time of position measurement
    Xp = Data.Xgps
    #value of position measurement
    Yp = Data.latlonToMeter(Data.Ygps)
    #time of acceleration measurement
    Xa = Integration.Xacc
    #value of acceleration measurement
    Ya = Integration.Yacc[:,1:]
    forward = Integration.my_integration(Xa, Integration.Yacc)[4]
    for i in range(Ya.shape[0]):
        Ya[i,:] = rotate(Ya[i,:], forward[i,:])
    
    x, y, err = derivative_GP(Xp, Xa, Yp, Ya)

    pyplot.figure()
    pyplot.plot(Xp, Yp, label="position data")
    pyplot.plot(x, y, label="prediction")
    pyplot.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y.flatten() - 1.9600 * np.diag(err),
                                (y.flatten() + 1.9600 * np.diag(err))[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
    pyplot.legend()
    