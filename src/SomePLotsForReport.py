#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 07:10:51 2018

@author: stephan
"""
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct, ExpSineSquared
import numpy as np
import Data
from myGP import derivative_GP
from vectorRotation import rotatedAcceleration
from SensorFusionGP import initialValues
#from sklearn.utils import check_random_state
from scipy.integrate import cumtrapz


def plot_1d_gp_gyr():
    n = 5000
    x_pred = np.atleast_2d(np.linspace(Data.Xgyr[0,:], Data.Xgyr[-1,:], n)).T
    kernel = ConstantKernel(0.2, (1e-7, 2)) * RBF(0.04, (1e-7, 1)) + WhiteKernel(0.1, (1e-7,0.2))#RBF(0.2, (0.01,0.1))
    
    fig = plt.figure(figsize=(7,9))
    fig.suptitle("1D homoscedastic GPs for each angular velocity dimension")
    
    ax = fig.add_subplot(311)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgyr, Data.Ygyr[:,Data.YAW])
    Ygyr, Sgyr = gp.predict(x_pred, True) #smoothes gyr. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Angular yaw velocity: $\sigma_{{rbf}}={0:.4f}, l_{{rbf}}={1:.4f}, noise={2:.5f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr-1.9600*Sgyr, (Ygyr+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.YAW], "r-", linewidth=.5, label="data")
    ax.plot(x_pred, Ygyr, "b-", label="estimate")
    ax.set_ylabel("$\omega_{yaw}$ in $rad / s$")
    
    ax = fig.add_subplot(312)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgyr, Data.Ygyr[:,Data.ROLL])
    Ygyr, Sgyr = gp.predict(x_pred, True) #smoothes gyr. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Angular roll velocity: $\sigma_{{rbf}}={0:.4f}, l_{{rbf}}={1:.4f}, noise={2:.5f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr-1.9600*Sgyr, (Ygyr+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.ROLL], "r-", linewidth=.5, label="data")
    ax.plot(x_pred, Ygyr, "b-", label="estimate")
    ax.set_ylabel("$\omega_{roll}$ in $rad / s$")
    
    ax = fig.add_subplot(313)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgyr, Data.Ygyr[:,Data.PITCH])
    Ygyr, Sgyr = gp.predict(x_pred, True) #smoothes gyr. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Angular pitch velocity: $\sigma_{{rbf}}={0:.4f}, l_{{rbf}}={1:.4f}, noise={2:.5f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr-1.9600*Sgyr, (Ygyr+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.PITCH], "r-", linewidth=.5, label="data")
    ax.plot(x_pred, Ygyr, "b-", label="estimate")
    ax.set_ylabel("$\omega_{pitch}$ in $rad / s$")
    ax.set_xlabel("time in $s$")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.show()
    
def plot_1d_gp_acc():
    n = 5000
    x_pred = np.atleast_2d(np.linspace(Data.Xacc[0,:], Data.Xacc[-1,:], n)).T
    kernel = ConstantKernel(10, (1e-2, 20)) * RBF(0.04, (1e-7, 1)) + WhiteKernel(0.1, (1e-7,1))#RBF(0.2, (0.01,0.1))
    
    fig = plt.figure(figsize=(7,9))
    fig.suptitle("1D homoscedastic GPs for each acceleration dimension")
    
    ax = fig.add_subplot(311)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xacc, Data.Yacc[:,Data.UP]-np.mean(Data.Yacc[:,Data.UP]))
    Yacc, Sacc = gp.predict(x_pred, True) #smoothes acc. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Upward acceleration: $\sigma_{{rbf}}={0:.2f}, l_{{rbf}}={1:.2f}, noise={2:.4f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Yacc+np.mean(Data.Yacc[:,Data.UP])-1.9600*Sacc,
                           (Yacc+np.mean(Data.Yacc[:,Data.UP])+1.9600*Sacc)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xacc, Data.Yacc[:,Data.UP], "r", linewidth=.5, label="data")
    ax.plot(x_pred, Yacc+np.mean(Data.Yacc[:,Data.UP]), "b", label="estimate")
    ax.set_ylabel("$a_{up}$ in $m / s^2$")
    
    ax = fig.add_subplot(312)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xacc, Data.Yacc[:,Data.FORWARD])
    Yacc, Sacc = gp.predict(x_pred, True) #smoothes acc. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Forward acceleration: $\sigma_{{rbf}}={0:.2f}, l_{{rbf}}={1:.2f}, noise={2:.4f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Yacc-1.9600*Sacc, (Yacc+1.9600*Sacc)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xacc, Data.Yacc[:,Data.FORWARD], "r", linewidth=.5, label="data")
    ax.plot(x_pred, Yacc, "b", label="estimate")
    ax.set_ylabel("$a_{forward}$ in $m / s^2$")
    
    ax = fig.add_subplot(313)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xacc, Data.Yacc[:,Data.LEFT])
    Yacc, Sacc = gp.predict(x_pred, True) #smoothes acc. data
    s_rbf = np.sqrt(gp.kernel_.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k2.get_params()['noise_level']
    ax.set_title("Leftward acceleration: $\sigma_{{rbf}}={0:.2f}, l_{{rbf}}={1:.2f}, noise={2:.4f}$".format(s_rbf, l_rbf, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Yacc-1.9600*Sacc, (Yacc+1.9600*Sacc)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xacc, Data.Yacc[:,Data.LEFT], "r", linewidth=.5, label="data")
    ax.plot(x_pred, Yacc, "b", label="estimate")
    ax.set_ylabel("$a_{left}$ in $m / s^2$")
    ax.set_xlabel("time in $s$")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.show()

def plot_1d_gp_gps():
    n = 5000
    YgpsLatLon = Data.latlonToMeter(Data.Ygps)
    x_pred = np.atleast_2d(np.linspace(Data.Xgps[0,:], Data.Xgps[-1,:], n)).T
    kernel = ConstantKernel(20, (1, 1000)) * RBF(1, (0.04, 10)) +\
        WhiteKernel(4, (.1,10)) + ConstantKernel(1,(1e-7,1))*DotProduct(1., (1,1))
    
    fig = plt.figure(figsize=(14,7))
    fig.suptitle("1D homoscedastic GPs for each GPS dimension")
    
    ax = fig.add_subplot(211)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgps, YgpsLatLon[:,0]-np.mean(YgpsLatLon[:,0]))
    Ygps, Sgps = gp.predict(x_pred, True) #smoothes gps. data
    print(gp.kernel_)
    s_rbf = np.sqrt(gp.kernel_.k1.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k1.k2.get_params()['noise_level']
    s_dot = np.sqrt(gp.kernel_.k2.k1.get_params()['constant_value'])
    ax.set_title("Latitude: $\sigma_{{rbf}}={0:.2f}, "
                 "l_{{rbf}}={1:.2f}, \sigma_{{dot}}={2:.4f}, noise={3:.4f}$".format(s_rbf, l_rbf, s_dot, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygps+np.mean(YgpsLatLon[:,0])-1.9600*Sgps,
                           (Ygps+np.mean(YgpsLatLon[:,0])+1.9600*Sgps)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xgps, YgpsLatLon[:,0], "rx", label="data")
    ax.plot(x_pred, Ygps+np.mean(YgpsLatLon[:,0]), "b", label="estimate")
    ax.set_ylabel(u"distance from start in $m$")
    
    ax = fig.add_subplot(212)
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgps, YgpsLatLon[:,1]-np.mean(YgpsLatLon[:,1]))
    Ygps, Sgps = gp.predict(x_pred, True) #smoothes gps. data
    print(gp.kernel_)
    s_rbf = np.sqrt(gp.kernel_.k1.k1.k1.get_params()['constant_value'])
    l_rbf = gp.kernel_.k1.k1.k2.get_params()['length_scale']
    noise = gp.kernel_.k1.k2.get_params()['noise_level']
    s_dot = np.sqrt(gp.kernel_.k2.k1.get_params()['constant_value'])
    ax.set_title("Longitude: $\sigma_{{rbf}}={0:.2f}, "
                 "l_{{rbf}}={1:.2f}, \sigma_{{dot}}={2:.4f}, noise={3:.4f}$".format(s_rbf, l_rbf, s_dot, noise))
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygps+np.mean(YgpsLatLon[:,1])-1.9600*Sgps,
                           (Ygps+np.mean(YgpsLatLon[:,1])+1.9600*Sgps)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.plot(Data.Xgps, YgpsLatLon[:,1], "rx", label="data")
    ax.plot(x_pred, Ygps+np.mean(YgpsLatLon[:,1]), "b", label="estimate")
    ax.set_ylabel(u"distance from start in $m$")
    ax.set_xlabel("time in $s$")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()

#---plot 2 (1D-GP)
def plot_3d_gp():
    n = 5000
    x_pred = np.atleast_2d(np.linspace(Data.Xgyr[0,:], Data.Xgyr[-1,:], n)).T
    kernel = ConstantKernel() * RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel, normalize_y=True)
    gp.fit(Data.Xgyr, Data.Ygyr)
    Ygyr, Sgyr = gp.predict(x_pred, True) #smoothes gyr. data
    
    fig = plt.figure()
    fig.suptitle("3D homoscedastic GPR of angular velocity\n"+\
                 str(gp.kernel_))
    
    ax = fig.add_subplot(311)
    ax.set_title("Angular yaw velocity")
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.YAW], "x", label="data")
    ax.plot(x_pred, Ygyr[:,Data.YAW], "b", label="estimate")
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr[:,Data.YAW]-1.9600*Sgyr, (Ygyr[:,Data.YAW]+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='b')
    ax.set_ylabel("Angular velocity in $° / s$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax = fig.add_subplot(312)
    ax.set_title("Angular roll velocity")
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.ROLL], "x", label="data")
    ax.plot(x_pred, Ygyr[:,Data.ROLL], "darkgreen", label="estimate")
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr[:,Data.ROLL]-1.9600*Sgyr, (Ygyr[:,Data.ROLL]+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='darkgreen')
    ax.set_ylabel("Angular velocity in $° / s$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax = fig.add_subplot(313)
    ax.set_title("Angular pitch velocity")
    ax.plot(Data.Xgyr, Data.Ygyr[:,Data.PITCH], "x", label="data")
    ax.plot(x_pred, Ygyr[:,Data.PITCH], "r", label="estimate")
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]), 
            np.concatenate([Ygyr[:,Data.PITCH]-1.9600*Sgyr, (Ygyr[:,Data.PITCH]+1.9600*Sgyr)[::-1]]),
            alpha=.3, ec='None', label="95% confidence interval", fc='r')
    ax.set_ylabel("Angular velocity in $° / s$")
    ax.set_xlabel("time in $s$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    fig.show()

def plot_derivative_gp_lsqf():
    X_pos = Data.Xgps#[1:,:]
    Ygps = Data.latlonToMeter(Data.Ygps)#[1:,:]
    X_acc = Data.Xacc_corrected2
    #use fitted data
    v0 = np.array([22.14242341, -2.17809353])
    p0 = np.array([18.97244865, -5.11213311])
    forward = np.array([23.83197455, -6.74910446])
    Yacc = Data.Yacc + [0, -0.42233391105525775, 0.5373903677236002]
    Yacc = rotatedAcceleration(X_acc, Yacc, v0[0], v0[1], forward)
    #Y_acc = Yacc[:,1:]
    #Y_pos = Ygps[:,:1]
    s1, l, s2, noiseA, noiseP = 20., 2., 1., 4, 4.
    s=0 #0 including starting gps point, 1 excluding starting gps point
    t, m, e = derivative_GP(X_pos[s:,:], X_acc, Ygps[s:,1:], Yacc[:,:1],
                            s1, l, s2, noiseP, noiseA)
    t2, m2, e2 = derivative_GP(X_pos[s:,:], X_acc, Ygps[s:,:1], Yacc[:,1:],
                            s1, l, s2, noiseP, noiseA)
    
    """ plotting """
    fig = plt.figure(figsize=(10,4))
    fig.suptitle("Linear Transformation GPR Fusion of GPS and least square fitted accelerometer data\n"
       "($\sigma_{{rbf}}={0}, l_{{rbf}}={1}, \sigma_{{dot}}={2}, noise_{{acc}}={3}, noise_{{gps}}={4}$)".format(
               s1, l, s2, noiseA, noiseP))
    ax = fig.add_subplot(121)
    ax.set_title("Longitudinal component")
    ax.plot(X_pos, Ygps[:,1:], "b.", label="GPS position")
    ax.plot(X_acc, cumtrapz(cumtrapz(Yacc[:,:1], X_acc, axis=0, initial=0)+v0[0],
                            X_acc, axis=0, initial=0)+p0[0],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax.fill_between(t.flatten(), m.flatten() + 1.9600*np.diag(e),
                                 m.flatten() - 1.9600*np.diag(e),
                    alpha=.3, facecolor='b', edgecolor='none',
                    label='95% confidence interval')
    ax.plot(t.flatten(), m.flatten(), "g", label="Estimte")
    ax.set_ylabel(u"Longitudinal distance from start in $m$")
    ax.set_xlabel("Time in $s$")
    ax.legend()
    ax2 = fig.add_subplot(122)
    ax2.set_title("Latitudinal component")
    ax2.plot(X_pos, Ygps[:,:1], "b.", label="GPS position")
    ax2.plot(X_acc, cumtrapz(cumtrapz(Yacc[:,1:], X_acc, axis=0, initial=0)+v0[1],
                            X_acc, axis=0, initial=0)+p0[1],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax2.fill_between(t2.flatten(), m2.flatten() + 1.9600*np.diag(e2),
                                   m2.flatten() - 1.9600*np.diag(e2),
                    alpha=.3, facecolor='b', edgecolor='none',
                    label='95% confidence interval')
    ax2.plot(t2.flatten(), m2.flatten(), "g", label="Estimte")
    ax2.set_ylabel(u"Latitudinal distance from start in $m$")
    ax2.set_xlabel("Time in $s$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.show()
    
def plot_derivative_gp():
    X_pos = Data.Xgps#[1:,:]
    Ygps = Data.latlonToMeter(Data.Ygps)#[1:,:]
    X_acc = Data.Xacc_corrected2
    #use fitted data
    v0 = np.array([22.14242341, -2.17809353])
    p0 = np.array([18.97244865, -5.11213311])
    forward = np.array([23.83197455, -6.74910446])
    Yacc = Data.Yacc + [0, -0.42233391105525775, 0.5373903677236002]
    Yacc_lsqf = rotatedAcceleration(X_acc, Yacc, v0[0], v0[1], forward)
    #p0 = v0 = np.array([0, 0])
    Yacc = rotatedAcceleration(X_acc, Data.Yacc, v0[0], v0[1], forward)
    #Y_acc = Yacc[:,1:]
    #Y_pos = Ygps[:,:1]
    s1, l, s2, noiseA, noiseP = 20., 2., 1., 4., 4.
    s=0 #0 including starting gps point, 1 excluding starting gps point
    t, m, e = derivative_GP(X_pos[s:,:], X_acc, Ygps[s:,1:], Yacc[:,:1],
                            s1, l, s2, noiseP, noiseA)
    t2, m2, e2 = derivative_GP(X_pos[s:,:], X_acc, Ygps[s:,:1], Yacc[:,1:],
                            s1, l, s2, noiseP, noiseA)
    
    """ plotting """
    fig = plt.figure(figsize=(10,4))
    fig.suptitle("Linear Transformation GPR Fusion of GPS and accelerometer data\n"
       "($\sigma_{{rbf}}={0}, l_{{rbf}}={1}, \sigma_{{dot}}={2}, noise_{{acc}}={3}, noise_{{gps}}={4}$)".format(
               s1, l, s2, noiseA, noiseP))
    ax = fig.add_subplot(121)
    ax.set_title("Longitudinal component")
    ax.plot(X_pos, Ygps[:,1:], "b.", label="GPS position")
    ax.plot(X_acc, cumtrapz(cumtrapz(Yacc_lsqf[:,:1], X_acc, axis=0, initial=0)+v0[0],
                            X_acc, axis=0, initial=0)+p0[0],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax.plot(X_acc, cumtrapz(cumtrapz(Yacc[:,:1], X_acc, axis=0, initial=0)+v0[0],
                            X_acc, axis=0, initial=0)+p0[0],
                   "--", color='violet', label="Int. acc.")
    ax.fill_between(t.flatten(), m.flatten() + 1.9600*np.diag(e),
                                 m.flatten() - 1.9600*np.diag(e),
                    alpha=.3, facecolor='b', edgecolor='none',
                    label='95% confidence interval')
    ax.plot(t.flatten(), m.flatten(), "g", label="Estimte")
    ax.set_ylabel(u"Longitudinal distance from start in $m$")
    ax.set_xlabel("Time in $s$")
    ax.legend()
    ax2 = fig.add_subplot(122)
    ax2.set_title("Latitudinal component")
    ax2.plot(X_pos, Ygps[:,:1], "b.", label="GPS position")
    ax2.plot(X_acc, cumtrapz(cumtrapz(Yacc_lsqf[:,1:], X_acc, axis=0, initial=0)+v0[1],
                            X_acc, axis=0, initial=0)+p0[1],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax2.plot(X_acc, cumtrapz(cumtrapz(Yacc[:,1:], X_acc, axis=0, initial=0)+v0[1],
                            X_acc, axis=0, initial=0)+p0[1],
                   "--", color='violet', label="Int. acc.")
    ax2.fill_between(t2.flatten(), m2.flatten() + 1.9600*np.diag(e2),
                                   m2.flatten() - 1.9600*np.diag(e2),
                    alpha=.3, facecolor='b', edgecolor='none',
                    label='95% confidence interval')
    ax2.plot(t2.flatten(), m2.flatten(), "g", label="Estimte")
    ax2.set_ylabel(u"Latitudinal distance from start in $m$")
    ax2.set_xlabel("Time in $s$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.show()

#plot_1d_gp_gyr()
#plot_1d_gp_acc()
#plot_1d_gp_gps()
#plot_3d_gp()
#plot_derivative_gp()
plot_derivative_gp_lsqf()