# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:39:18 2018
@author: Stephan
"""

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider
from scipy.spatial.distance import pdist, squareform
from bisect import bisect_left
from math import pi
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
import Data

n = 1000

#Squared Exponential Kernel
def sqared_exponential_kernel(a, b, s=1., l=1.):
    """
    Squared Exponential Kernel for d dimensional vectors a and b. Each row
    of a and b represent a point in an d dimensional space. Returns a
    matrix with shape (len(b), len(a)):
    
    s² * exp(-1/(2 * l²) * sqdist(a, b))
    """
    #distance of two vectors with n dimensional points
    #from scipy.spatial import distance
    #sqdist = distance.cdist(a, b, 'sqeuclidean')
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2,1) - 2.*(a @ b.T)
    return (s**2) * np.exp(-1./(2. * l**2) * sqdist)

#Linear Kernel
def linear_kernel(x1, x2, s=1., c=0.):
    """
    Squared Exponential Kernel.
    
    c is the starting point for the linear prior.
    """
    return (s**2) * ((x1-c) * ((x2-c).T))

def kernel(a, b, s=1., l=1., s_linear=1.):
    """
    Currently used all purpose kernel.
    """
    return sqared_exponential_kernel(a, b, s, l) + linear_kernel(a, b, s=s_linear)

#for 1 dimension at a time!
def derivative_GP(X_pos, X_acc, Y_pos, Y_acc, s1, l, s2, noiseGPS = 4., noiseAcc = 0.2):
    """Uses asynchronous measurements of the data Y_pos together with its 2nd derivative Y_acc for GPR."""
    #X_pred = [X_pos X_acc]
    X_pred = np.concatenate((X_pos, X_acc))
    sort = X_pred[:,0].argsort()
    unique = np.ones(len(X_pred), dtype=bool)
    unique2 = np.ones(len(X_pred), dtype=bool)
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique[1:])
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique2[:-1])
    
    #test = X_pred[sort][unique]
    #training = np.concatenate((X_pos, X_acc))
    test = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), n)).T
    training = X_pred[sort][unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    dt = training[1:]-training[:-1]
    median_dt = np.median(dt)
    #---training = np.r_[training, [training[-1]+median_dt], [training[-1]+median_dt*2.0]]
    
    #Set the kernel which chooses the observed values from latent variables.
    #choose Kp such that Y_pos = Kp @ latentY(X_pred)
    Kp = np.zeros((len(X_pos), len(X_pred)))
    for i in range(Kp.shape[1]):
        if sort[i] < len(X_pos):
            Kp[sort[i], i] = 1.
    Kp = Kp[:,unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    #---Kp = np.c_[Kp, np.zeros((len(X_pos),2))]
    
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
    S2 = kernel(test, test, s1, l, s2)
    S12 = kernel(test, training, s1, l, s2)
    S21 = kernel(training, test, s1, l, s2)
    noise = [noiseGPS*noiseGPS]*len(Y_pos)+[noiseAcc*noiseAcc]*len(Y_acc)
    #noise[0] = 0
    mu2_1 = S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    S2_1 = S2 - S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ K @ S21
    #resuts from setting K2 to I [Shimin Feng 2014]
    #mu2_1 = I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    #S2_1 = S2 - I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T) @ K @ S21 @ I.T
    #in the original equations [Murray-Smith 2005]
    #mu2_1 = K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ M1
    #S2_1 = S2 - K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ K1 @ S21 @ K2
    return test, mu2_1, S2_1

def asynchronous_GP(X_pos, X_pos2, Y_pos, Y_pos2, s, l, s2, noisePos1 = 4., noisePos2 = 0.2):
    """uses two asynchronous measurements Y_pos and Y_pos2 for GPR."""
    #X_pred = [X_pos X_acc]
    X_pred = np.concatenate((X_pos, X_pos2))
    sort = X_pred[:,0].argsort()
    #unique = np.ones(len(X_pred), dtype=bool)
    #np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique[1:])
    
    x_star = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), n)).T
    x = X_pred[sort]#[unique]
    f = np.concatenate((Y_pos, Y_pos2))[sort]#[unique]
    if Y_pos.ndim == 1:
        f = np.atleast_2d(f).T
    
    #prepare noise in the same way as x and f, but allow homogenous noise
    if hasattr(noisePos1, "__len__") and len(noisePos1) == len(X_pos):
        noise = noisePos1
    else:
        noise = [noisePos1]*len(X_pos)
    if hasattr(noisePos2, "__len__") and len(noisePos2) == len(X_pos2):
        noise = np.concatenate((noise, noisePos2))
    else:
        noise = np.concatenate((noise, [noisePos2]*len(X_pos2)))
    noise = np.array(noise)[sort]#[unique]
    noise = np.square(np.diag(noise))
    
    K = kernel(x, x, s, l, s2)
    K_star = kernel(x, x_star, s, l, s2) #don't need to add noise since only the diagonal of K from f~GP(m, K) is effected by noise
    K_star_star = kernel(x_star, x_star, s, l, s2)
    mu_star = mu(x_star) + K_star.T @ np.linalg.inv(K + noise) @ (f - mu(x))
    #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K + noise) @ K_star
    return x_star, mu_star, sigma_star

def asynchronous_GP_using_latent_variables(X_pos, X_pos2, Y_pos, Y_pos2, s, l, s2, noiseGPS = 4., noiseAcc = 0.2):
    """uses two asynchronous measurements Y_pos and Y_pos2 for GPR."""
    #X_pred = [X_pos X_acc]
    X_pred = np.concatenate((X_pos, X_pos2))
    sort = X_pred[:,0].argsort()
    unique = np.ones(len(X_pred), dtype=bool)
    unique2 = np.ones(len(X_pred), dtype=bool)
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique[1:])
    np.not_equal(X_pred[sort][1:,0], X_pred[sort][:-1,0], out=unique2[:-1])
    
    #test = X_pred[sort][unique]
    #training = np.concatenate((X_pos, X_acc))
    test = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), n)).T
    training = X_pred[sort][unique]
    
    #Set the kernel which chooses the observed values from latent variables.
    #choose Kp1 such that Y_pos = Kp1 @ latentY(X_pred)
    Kp1 = np.zeros((len(X_pos), len(X_pred)))
    for i in range(Kp1.shape[1]):
        if sort[i] < len(X_pos):
            Kp1[sort[i], i] = 1.
    Kp1 = Kp1[:,unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    Kp1 = np.c_[Kp1, np.zeros((len(X_pos),2))]
    
    #choose Kp2 such that Y_acc = K2 @ latentY(X_pred)
    Kp2 = np.zeros((len(X_pos2), len(X_pred)))
    for i in range(Kp2.shape[1]):
        if sort[i] < len(X_pos2):
            Kp2[sort[i], i] = 1.
    Kp2 = Kp2[:,unique2]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    Kp2 = np.c_[Kp2, np.zeros((len(X_pos2),2))]
    
    #M = [Y_pos Y_acc]
    M = np.concatenate((Y_pos, Y_pos2))
    #K = [Kp Ka]
    K = np.concatenate((Kp1, Kp2))
    
    #now we have set M = K @ latentY and can try to find this latent Y
    S = kernel(training, training, s, l, s2)
    S2 = kernel(test, test, s, l, s2)
    S12 = kernel(test, training, s, l, s2)
    S21 = kernel(training, test, s, l, s2)
    noise = [noiseGPS*noiseGPS]*len(Y_pos)+[noiseAcc*noiseAcc]*len(Y_pos2)
    mu2_1 = S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    S2_1 = S2 - S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ K @ S21
    #resuts from setting K2 to I [Shimin Feng 2014]
    #mu2_1 = I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    #S2_1 = S2 - I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T) @ K @ S21 @ I.T
    #in the original equations [Murray-Smith 2005]
    #mu2_1 = K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ M1
    #S2_1 = S2 - K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ K1 @ S21 @ K2
    return test, mu2_1, S2_1


def mu(x):
    X = np.atleast_2d(x)
    return np.zeros(X.shape)

def mymu(x_star, x, f):
    if np.array_equal(x_star, x):
        return np.atleast_2d(f)
    else:
        #in Treppenmanier die Werte von f fortsetzen für x_star, solange
        #bis f einen neuen wert hat
        m = np.zeros(np.atleast_2d(x_star).shape)
        stepBegin = 0
        step = -1
        for step in range(0, len(x.flatten())-1):
            stepEnd = stepBegin+1
            m[stepBegin,0] = f[step,0]
            while x_star[stepEnd,0] < x[step+1,0]:
                m[stepEnd,0] = f[step,0]
                stepEnd += 1
            stepBegin = stepEnd
        for i in range(stepBegin, len(x_star.flatten())):
            m[i] = f[step+1, 0]
        return np.atleast_2d(m)
    
def gp(x=np.array([[-4.0],[0.0],[1.0],[2.0],[3.0]]),
            f=np.array([[-2],[1],[2],[2],[0.5]]),
            s=3, l=10, noise=1, x_star=np.linspace(-4., 3., n).reshape(-1, 1)):
    K = kernel(x, x, s, l)+np.eye(len(x))*noise
    K_star = kernel(x, x_star, s, l) #don't need to add noise since only the diagonal of K from f~GP(m, K) is effected by noise
    K_star_star = kernel(x_star, x_star, s, l)+np.eye(len(x_star))*noise
    mu_star = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f - mu(x))
    #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
    return mu_star, sigma_star

if __name__ == "__main__":
    def TEST(indexCurve=5, curve=-1.5, s = 2., l = 1.0, noiseP = 0.1, noiseA = 1., ax=None):
        if ax is None:
            ax = pyplot.figure().add_subplot(111)  
        t_p = np.linspace(0., 5., 5).reshape(-1,1)
        p = 2*t_p + 2*np.sin(t_p)
        t_a = np.linspace(0., 5., 125).reshape(-1,1)
        a = - 2*np.sin(t_a)
        a[indexCurve,0] = curve # falsch
        t, mu, sigma = derivative_GP(X_pos=t_p, Y_pos=p, X_acc=t_a, Y_acc=a,
                               s1 = s,
                               l = l,
                               s2 = 1.,
                               noiseGPS = noiseP,
                               noiseAcc = noiseA)
        
        
        ax.plot(t, mu, label="prediction")
        ax.plot(t_p, p, label="data")
        for i in range(0, len(a), int(np.ceil(len(a)/10))):
            j = bisect_left(t, t_a[i])
            ax.plot([t[j] , t[j]  + 0.1    ],
                    [mu[j], mu[j] + a[i]*0.1], "y-")
        j = bisect_left(t, t_a[indexCurve])
        ax.plot([t[j] , t[j]  + 0.1    ],
                [mu[j], mu[j] + a[indexCurve]*0.1], "r-", label="adjustable curvature")
        ax.fill(np.concatenate([t, t[::-1]]),
                         np.concatenate([mu.flatten() - 1.9600 * np.diag(sigma),
                                        (mu.flatten() + 1.9600 * np.diag(sigma))[::-1]]),
                         alpha=.3, fc='b', ec='None', label='95% confidence interval')
        ax.legend()
        return ax
    
    #---1st plot (illustrates the algorithm with some adjustable parameters)
    curve=-1.5
    s = 2.
    l = 0.3
    noiseP = 1.
    noiseA = 1.
    fig = pyplot.figure()
    fig.subplots_adjust(bottom=0.4)
    #axTEST = fig.add_subplot(111, projection='3d')
    axTEST = fig.add_subplot(111)
    title = "Illustrating GP fusion of asynchonous samples from a function\n"+\
            "$f(x) = x/4 + \sin (x)$ with its second derivative"
    axTEST.set_title(title)
    axicurve = fig.add_axes([0.38,0.21,0.50,0.03])
    axcurve  = fig.add_axes([0.38,0.17,0.50,0.03])
    axsigma  = fig.add_axes([0.38,0.13,0.50,0.03])
    axlength = fig.add_axes([0.38,0.09,0.50,0.03])
    axnoiseP = fig.add_axes([0.38,0.05,0.50,0.03])
    axnoiseA = fig.add_axes([0.38,0.01,0.50,0.03])
    sicurve = Slider(axicurve, 'index of 2nd deriv.',valmin=0, valmax=124, valinit=5, valfmt='%0i')
    scurve  = Slider(axcurve,  '2nd derivative',     valmin=-100, valmax=100, valinit=curve, valfmt='%0.2f')
    ssigma  = Slider(axsigma,  'sigma',              valmin=0.01, valmax=10, valinit=s, valfmt='%0.3f')
    slength = Slider(axlength, 'length scale',       valmin=0.01, valmax=1.0, valinit=l, valfmt='%0.4f')
    snoiseP = Slider(axnoiseP, 'noise of data',      valmin=0.0, valmax=10, valinit=noiseP, valfmt='%0.5f')
    snoiseA = Slider(axnoiseA, 'noise of curvature', valmin=0.0, valmax=10, valinit=noiseA, valfmt='%0.5f')
    TEST(5,curve, s, l, noiseP, noiseA, axTEST)
    axTEST.set_ylim(-3, 12)
    def updateTEST(val):
        axTEST.clear()
        axTEST.set_title(title)
        TEST(int(sicurve.val), scurve.val, ssigma.val, slength.val, snoiseP.val, snoiseA.val, axTEST)
        axTEST.set_ylim(-3, 12)
        fig.canvas.draw_idle()
    sicurve.on_changed(updateTEST)
    scurve.on_changed(updateTEST)
    ssigma.on_changed(updateTEST)
    slength.on_changed(updateTEST)
    snoiseP.on_changed(updateTEST)
    snoiseA.on_changed(updateTEST)

    def plot_derivative_gp(t_p = np.array([[.0],[.1],[1.5],[2.]]),
                           p   = np.array([[.0],[.5],[-1.5],[0.5]]),
                           t_a = np.array([[0.],[1.1],[2.]]),
                           a   = np.array([[1.],[1.5],[.5]]),
                           s = 8.,
                           l = 2.,
                           s2 = 1.,
                           noiseGPS = 4.,
                           noiseAcc = 4.,
                           ax=None):
        t, lat, err_lat = derivative_GP(t_p, t_a, p[:,0].reshape(-1,1), a[:,0].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
        t, lon, err_lon = derivative_GP(t_p, t_a, p[:,1].reshape(-1,1), a[:,1].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
        lat = lat.flatten()
        lon = lon.flatten()
        if ax is None:
            ax = pyplot.figure().add_subplot(111)
        ax.plot(lon, lat, label='pred')
        ax.plot(p[:,1], p[:,0], ".", label='data')
        ax.fill(np.concatenate([lon, lon[::-1]]),
                     np.concatenate([lat - 1.9600 * np.diag(err_lat),
                                    (lat + 1.9600 * np.diag(err_lon))[::-1]]),
                     alpha=.3, fc='b', ec='None', label='95% confidence interval')
        ax.legend()
    
    
    #---2nd plot (fitted derivative gp)
    #[22.13850709, -2.16845578, -0.42172239,  0.53603343, 23.83197455, -6.74910446]
    #(18.97244945, -5.11213289, #start position)
    #set up acceleration according to the least square fit
    calibratedYacc = Data.Yacc + [0, -0.42172238739533024, 0.5360334305256229]
    from vectorRotation import rotatedAcceleration
    acc = rotatedAcceleration(Data.Xacc, calibratedYacc,
                                22.13850709, -2.16845578, #start speed
                                np.array([23.83197455, -6.74910446]))
    gps = Data.latlonToMeter(Data.Ygps)
    s = 20.
    l = 2.
    s2 = 1.
    noiseGPS = 4.
    noiseAcc = 4.
    t, lat, err_lat = derivative_GP(Data.Xgps, Data.Xacc, gps[:,0].reshape(-1,1), acc[:,0].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
    t, lon, err_lon = derivative_GP(Data.Xgps, Data.Xacc, gps[:,1].reshape(-1,1), acc[:,1].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
    lat = lat.flatten()
    lon = lon.flatten()
    fig = pyplot.figure(figsize=(6,4))
    fig.suptitle("Fusion with linear transformation GP\n"+\
                 "($\sigma{{rbf}}={0:0.1f}, l_{{rbf}}={1:0.1f}, \sigma{{dot}}={2:0.1f}, noise_{{gps}}={3:0.1f}, noise_{{acc}}={4:0.1f}$)".format(
                                    s,                  l,                      s2,                 noiseGPS,                   noiseAcc))
    ax = fig.add_subplot(211)
    ax.set_title("Longitude")
    ax.plot(Data.Xgps[:,0], gps[:,1], ".", label='GPS data')
    ax.plot(t, lon, label='Estimate')
    ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([lon - 1.9600 * np.diag(err_lon),
                                (lon + 1.9600 * np.diag(err_lon))[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
    ax.set_xlabel("Time in $s$")
    ax.set_ylabel("Position in $m$")
    ax = fig.add_subplot(212)
    ax.set_title("Latitude")
    ax.plot(Data.Xgps[:,0], gps[:,0], ".", label='GPS data')
    ax.plot(t, lat, label='Estimate')
    ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([lat - 1.9600 * np.diag(err_lon),
                                (lat + 1.9600 * np.diag(err_lon))[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
    ax.set_xlabel("Time in $s$")
    ax.set_ylabel("Position in $m$")
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.show()
    
    
    #plots samples drawn from the prior defined by an RBF kernel
    def plot_gp_prio_samples(s=5, l=0.1):
        Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
        K_ = sqared_exponential_kernel(Xtest, Xtest, s, l)
        L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
        #f_prior = np.dot(L, np.random.normal(size=(n,10)))
        f_prior = L @ np.random.normal(size=(n,10))    
        pyplot.plot(Xtest,f_prior)
        pyplot.ylim(-50, 50)
        pyplot.title('Squared Exponential Kernel\n(sigma={}, length_scale={})'.format(s, l))


    def plot_gp_prio_samples_some_parameters():
        
        pyplot.figure()
        pyplot.subplot(221)
        plot_gp_prio_samples(20, 2)
        pyplot.subplot(222)
        plot_gp_prio_samples(20, 1)
        pyplot.subplot(223)
        plot_gp_prio_samples(5, 2)
        pyplot.subplot(224)
        plot_gp_prio_samples(5, 1)
        pyplot.show()
    
    def plot_gp(x=np.array([[-4.0],[0.0],[1.0],[2.0],[3.0]]),
                f=np.array([[-2],[1],[2],[2],[0.5]]), ndata=1,
                s=3, l=10, noise=1, ax=None, normalize=True):
        if ax is None:
            ax = pyplot.figure().add_subplot(111)
        meanf = None
        stdf = None
        minx = min(x)
        maxx = max(x)
        minf = min(f)
        maxf = max(f)
        if normalize:
            meanf = np.mean(f, axis=0)
            stdf = np.std(f, axis=0)
            f = (f-meanf)/stdf
        x=x[:ndata,:]
        f=f[:ndata,:]
        #x_star = np.linspace(minx-0.1*abs(minx), maxx+0.1*abs(maxx), n).reshape(-1, 1)
        x_star = np.linspace(minx, maxx, n).reshape(-1, 1)
        
        mu_star, sigma_star = gp(x, f, s, l, noise, x_star)
        
        if normalize:
            mu_star = mu_star*stdf + meanf
            f = f*stdf + meanf
        sigma_star = np.diagonal(sigma_star).reshape(-1,1)#test
        ax.plot(x_star, mu_star, label='prediction')
        ax.fill(np.concatenate([x_star, x_star[::-1]]),
                 np.concatenate([mu_star - 1.9600 * sigma_star,
                                (mu_star + 1.9600 * sigma_star)[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
        ax.plot(x, f, ".", label='data')
        ax.set_ylim(minf-0.3*abs(maxf-minf),maxf+0.3*abs(maxf-minf))
        ax.set_title('Posterior transformed by data (slider)')
    
    def plot_adjustable_GP():
        fig = pyplot.figure()
        fig.subplots_adjust(bottom=0.2)
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        axdata = fig.add_axes([0.38,0.14,0.50,0.03])
        axsigma = fig.add_axes([0.38,0.09,0.50,0.03])
        axlength = fig.add_axes([0.38,0.05,0.50,0.03])
        axnoise = fig.add_axes([0.38,0.01,0.50,0.03])
        gps = Data.latlonToMeter(Data.Ygps)#Data.Ygps
        gpsx = gps[:,0].reshape(-1,1) #lat
        gpsy = gps[:,1].reshape(-1,1) #lon
        
        #==============================================================================
        # s = 90000
        # l = 60
        # noise=9
        #==============================================================================
        s=20
        l=1.5
        noise=4
        ndata=len(gpsx)
        sdata = Slider(axdata, 'number of data points', valmin=1, valmax=len(gpsx), valinit=len(gpsx), valfmt='%0.0f')
        ssigma = Slider(axsigma, 'sigma', valmin=0.01, valmax=100, valinit=s, valfmt='%0.3f')
        slength = Slider(axlength, 'length scale', valmin=0.1, valmax=100.0, valinit=l, valfmt='%0.4f')
        snoise = Slider(axnoise, 'noise', valmin=1e-7, valmax=100, valinit=noise, valfmt='%0.5f')
        plot_gp(Data.Xgps, gpsx, ndata, s, l, noise, ax)
        #plot_gp2(Data.Xgps, gps, len(gpsx), 90000, 60, 9, ax)
        def update(val):
            ndata = int(round(sdata.val))
            s = ssigma.val
            l = slength.val
            noise = snoise.val
            ax.clear() #workaround becaue I could not update the "fill" part
            plot_gp(Data.Xgps, gpsx, ndata, s, l, noise, ax)
            #plot_gp2(Data.Xgps, gps, ndata, s, l, noise, ax)
            fig.canvas.draw_idle()
        sdata.on_changed(update)
        ssigma.on_changed(update)
        slength.on_changed(update)
        snoise.on_changed(update)
        return [sdata, ssigma, slength, snoise]
    
    
    def plot_adjustable_derivative_GP():
        fig = pyplot.figure()
        fig.subplots_adjust(bottom=0.29)
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        title = "GPR Fusion of GPS data with Accelerometer data as Derivative"
        ax.set_title(title)
        axdata   = fig.add_axes([0.38,0.21,0.50,0.03])
        axsigma  = fig.add_axes([0.38,0.17,0.50,0.03])
        axsigma2 = fig.add_axes([0.38,0.13,0.50,0.03])
        axlength = fig.add_axes([0.38,0.09,0.50,0.03])
        axnoise  = fig.add_axes([0.38,0.05,0.50,0.03])
        axnoise2 = fig.add_axes([0.38,0.01,0.50,0.03])
        gps = Data.latlonToMeter(Data.Ygps)#Data.
                
        #gpsy = gps[:,0].reshape(-1,1) #lat
        #gpsx = gps[:,1].reshape(-1,1) #lon
        
        from vectorRotation import rotatedAcceleration
        from SensorFusionGP import initialValues
        v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, gps[:,0], gps[:,1])
        ar = rotatedAcceleration(Data.Xacc, Data.Yacc, v0[0], v0[1], forward)
        #==============================================================================
        # s = 90000
        # l = 60
        # noise=9
        #==============================================================================
        s=20
        l=1.5
        s2=1.0
        noise=4
        #ndata=len(gpsx)
        sdata     = Slider(axdata,  'number of data points', valmin=1, valmax=len(ar), valinit=len(ar), valfmt='%0.0f')
        ssigma    = Slider(axsigma, 'sigma rbf', valmin=1., valmax=200., valinit=s, valfmt='%0.3f')
        ssigma2   = Slider(axsigma2,'sigma dot', valmin=0.0, valmax=10., valinit=s2, valfmt='%0.3f')
        slength   = Slider(axlength,'length scale', valmin=0.01, valmax=20, valinit=l, valfmt='%0.4f')
        snoise    = Slider(axnoise, 'noise acc', valmin=0.0, valmax=10, valinit=noise, valfmt='%0.5f')
        snoise_gps= Slider(axnoise2,'noise gps', valmin=0.0, valmax=10, valinit=noise, valfmt='%0.5f')
        #plot_gp(Data.Xgps, gpsy, ndata, s, l, noise, ax)
        plot_derivative_gp(Data.Xgps, gps, Data.Xacc, ar, s, l, s2, 4, noise, ax)
        ax.set_xlim(np.min(gps[:,1])-5, np.max(gps[:,1])+5)
        ax.set_ylim(np.min(gps[:,0])-5, np.max(gps[:,0])+5)
        #plot_gp2(Data.Xgps, gps, len(gpsx), 90000, 60, 9, ax)
        def update(val):
            ndata = int(round(sdata.val))
            s = ssigma.val
            l = slength.val
            s2 = ssigma2.val
            noise = snoise.val
            noise_gps = snoise_gps.val
            ax.clear() #workaround becaue I could not update the "fill" part
            ax.set_title(title)
            #plot_gp(Data.Xgps, gpsy, ndata, s, l, noise, ax)
            plot_derivative_gp(Data.Xgps, gps, Data.Xacc[:ndata], ar[:ndata,:], s, l, s2, noise_gps, noise, ax)
            ax.set_xlim(np.min(gps[:,1])-5, np.max(gps[:,1])+5)
            ax.set_ylim(np.min(gps[:,0])-5, np.max(gps[:,0])+5)
            #plot_gp2(Data.Xgps, gps, ndata, s, l, noise, ax)
            fig.canvas.draw_idle()
        sdata.on_changed(update)
        ssigma.on_changed(update)
        ssigma2.on_changed(update)
        slength.on_changed(update)
        snoise.on_changed(update)
        snoise_gps.on_changed(update)
        return [sdata, ssigma, slength, snoise, snoise_gps]
    
    def plot_gp2(x=np.array([[0.0],[1.04],[1.08],[1.12],[3.0]]),
                f=np.array([[0,0],[0.1,-0.1],[1,0],[3,1],[5,1]]), ndata=1,
                s=3.0, l=10.0, noise=0, ax=None, normalize=True, dataPoints=None,
                predictionLine=None):
        if ax == None:
            fig= pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')
            fig.subplots_adjust(bottom=0.2)
        meanf = None
        stdf = None
        minx = min(x)
        maxx = max(x)
        minf = np.min(f, axis=0)
        maxf = np.max(f, axis=0)
        if normalize:
            meanf = np.mean(f, axis=0)
            stdf = np.std(f, axis=0)
            f = (f-meanf)/stdf
        x=x[:ndata,:]
        f=f[:ndata,:]
        f1 = np.atleast_2d(f[:,0]).T
        f2 = np.atleast_2d(f[:,1]).T
        K = kernel(x, x, s, l, s)+np.eye(ndata)*noise
        #x_star = np.linspace(minx-0.1*abs(minx), maxx+0.1*abs(maxx), n).reshape(-1, 1)
        x_star = np.linspace(minx, maxx, n).reshape(-1, 1)
        K_star = kernel(x, x_star, s, l, s)
        K_star_star = kernel(x_star, x_star, s, l, s)+np.eye(n)*noise
        mu_star1 = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f1 - mu(x))
        mu_star2 = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f2 - mu(x))
        #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
        if normalize:
            mu_star1 = mu_star1*stdf[0] + meanf[0]
            mu_star2 = mu_star2*stdf[1] + meanf[1]
            f = f*stdf + meanf
        sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
        sigma_star = np.diagonal(sigma_star).reshape(-1,1)#test
        
        if (dataPoints is None) or (predictionLine is None):
            """plot"""
            dataPoints = ax.plot(x, f[:,0], f[:,1], ".", label='data')
            predictionLine = ax.plot(x_star.flatten(), mu_star1.flatten(), mu_star2.flatten(), "-", label='prediction')    
            
        #==============================================================================
        #     ax.fill(np.concatenate([mu_star1, mu_star1[::-1]]),
        #              np.concatenate([mu_star2 - 1.9600 * sigma_star,
        #                             (mu_star2 + 1.9600 * sigma_star)[::-1]]),
        #              alpha=.3, fc='b', ec='None', label='95% confidence interval')
        #==============================================================================
            
            ax.set_xlim(minx[0]-0.3*abs(maxx[0]-minx[0]),maxx[0]+0.3*abs(maxx[0]-minx[0]))
            ax.set_ylim(minf[0]-0.3*abs(maxf[0]-minf[0]),maxf[0]+0.3*abs(maxf[0]-minf[0]))
            ax.set_zlim(minf[1]-0.3*abs(maxf[1]-minf[1]),maxf[1]+0.3*abs(maxf[1]-minf[1]))
            ax.invert_yaxis()
            ax.set_xlabel(u'time in $s$')
            ax.set_ylabel(u'latitude in $°$')
            ax.set_zlabel(u'longitude in $°$')
            ax.legend()
            ax.set_title('Posterior transformed by data (slider)')
            return dataPoints, predictionLine
        else:
            #dataPoints[0].remove()
            dataPoints[0].set_data(x.flatten(), f[:,0])
            dataPoints[0].set_3d_properties(f[:,1])
            #predictionLine[0].remove()
            predictionLine[0].set_data(x_star.flatten(), mu_star1.flatten())
            predictionLine[0].set_3d_properties(mu_star2.flatten())
            return dataPoints, predictionLine
    
    #---3nd plot (samples of prior)
    plot_gp_prio_samples_some_parameters()
    #---4rd plot (adjustable GP)
    preserveSliders = plot_adjustable_GP()
    #---5th plot (adjustable derivative GP)
    preserveSliders2 = plot_adjustable_derivative_GP()

    
    #---6th plot
    fig2 = pyplot.figure()
    fig2.subplots_adjust(bottom=0.2)
    ax2 = fig2.add_subplot(111, projection='3d')
    #ax = fig2.add_subplot(111)
    axdata2 = fig2.add_axes([0.38,0.14,0.50,0.03])
    axsigma2 = fig2.add_axes([0.38,0.09,0.50,0.03])
    axlength2 = fig2.add_axes([0.38,0.05,0.50,0.03])
    axnoise2 = fig2.add_axes([0.38,0.01,0.50,0.03])
    gps = Data.Ygps#Data.latlonToMeter(Data.Ygps)
    gpsx = gps[:,0].reshape(-1,1) #lat
    gpsy = gps[:,1].reshape(-1,1) #lon
    #==============================================================================
    # s = 90000
    # l = 60
    # noise=9
    #==============================================================================
    s=0.1
    l=10
    noise=0.0001
    sdata2 = Slider(axdata2, 'number of data points', valmin=1, valmax=len(gpsx), valinit=len(gpsx), valfmt='%0.0f')
    ssigma2 = Slider(axsigma2, 'sigma', valmin=0.01, valmax=100, valinit=s, valfmt='%0.3f')
    slength2 = Slider(axlength2, 'length scale', valmin=1, valmax=100.0, valinit=l, valfmt='%0.4f')
    snoise2 = Slider(axnoise2, 'noise', valmin=0.000001, valmax=10, valinit=noise, valfmt='%0.5f')
    #plot_gp(Data.Xgps, gpsy, len(gpsx), s, l, noise, ax)
    datap, positionl = plot_gp2(Data.Xgps, gps, len(gpsx), s, l, noise, ax2)
    def update2(val):
        ndata = int(round(sdata2.val))
        s = ssigma2.val
        l = slength2.val
        noise = snoise2.val
        #ax.clear() #workaround becaue I could not update the "fill" part
        #plot_gp(Data.Xgps, gpsy, ndata, s, l, noise, ax)
        global datap
        global positionl
        datap, positionl = plot_gp2(Data.Xgps, gps, ndata, s, l, noise, ax2, True, datap, positionl)
        fig2.canvas.draw_idle()
    sdata2.on_changed(update2)
    ssigma2.on_changed(update2)
    slength2.on_changed(update2)
    snoise2.on_changed(update2)
    
    #---7th plot
    #n=10000
    from SensorFusionGP import initialValues
    from Integration import rotatingIntegration
    #prepare position data
    p_gps = Data.latlonToMeter(Data.Ygps)[:,::-1]
    #prepare acceleration data
    v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, p_gps[:,0], p_gps[:,1])
    p_acc_x, p_acc_y = rotatingIntegration(Data.Xacc, Data.Yacc, 0, 0, v0[1], v0[0], forward[::-1])
    p_acc = np.array([p_acc_x, p_acc_y]).T
    #set hyperparameters
    s_rbf     = 20. #20
    l_rbf     = 2.  #2
    s_dot     = 1.  #1
    noise_acc = 50. #50
    noise_gps = 4.  #4   10  
    forTesting = True
    #GPR
    t, m, s = asynchronous_GP(Data.Xacc, Data.Xgps, p_acc, p_gps,
                           s_rbf, l_rbf, s_dot, noise_acc, noise_gps)
    #plotting
    fig = pyplot.figure()
    fig.subplots_adjust(bottom=0.25)
    if forTesting:
        axsrbf      = fig.add_axes([0.31,0.17,0.52,0.03])
        axlrbf      = fig.add_axes([0.31,0.13,0.52,0.03])
        axsdot      = fig.add_axes([0.31,0.09,0.52,0.03])
        axnoise_acc = fig.add_axes([0.31,0.05,0.52,0.03])
        axnoise_gps = fig.add_axes([0.31,0.01,0.52,0.03])
        ssrbf      = Slider(axsrbf, 'sigma rbf', valmin=0.1, valmax=200., valinit=s_rbf, valfmt='%0.0f')
        slrbf      = Slider(axlrbf, 'length scale rbf', valmin=0.01, valmax=20, valinit=l_rbf, valfmt='%0.1f')
        ssdot      = Slider(axsdot, 'sigma dot', valmin=0.01, valmax=10., valinit=s_dot, valfmt='%0.2f')
        snoise_acc = Slider(axnoise_acc, 'noise acc', valmin=0.0, valmax=500, valinit=noise_acc, valfmt='%0.0f')
        snoise_gps = Slider(axnoise_gps, 'noise gps', valmin=0.0, valmax=50, valinit=noise_gps, valfmt='%0.1f')
    ax = fig.add_subplot(111)
    ax.set_title("GP Fusion of GPS and integrated accelerometer data")
    ax.plot(p_acc[:,0], p_acc[:,1], label="Integrated acceleration")
    ax.plot(p_gps[:,0], p_gps[:,1], label="GPS position")
    l_m = ax.plot(m[:,0], m[:,1], label="Estimate")[0]
# =============================================================================
#     l_sy = ax.fill(np.concatenate([m[:,0], m[::-1,0]]),
#                   np.concatenate([m[:,1] - np.sqrt(np.diag(s)),
#                                  (m[:,1] + np.sqrt(np.diag(s)))[::-1]]),
#                   alpha=.3, fc='b', ec='None', label='GPR variance $\sigma$')[0]
# =============================================================================
# =============================================================================
#     l_sy = ax.fill_between(m[:,0], m[:,1] - np.sqrt(np.diag(s)), m[:,1] + np.sqrt(np.diag(s)),
#                            alpha=.3, color='b', edgecolor='None', label='GPR variance $\sigma$')
# =============================================================================
# =============================================================================
#     l_sx = ax.fill(np.concatenate([m[:,0] - np.sqrt(np.diag(s)),
#                                   (m[:,0] + np.sqrt(np.diag(s)))[::-1]]),
#                   np.concatenate([m[:,1], m[:,1][::-1]]),
#                   alpha=.3, fc='b', ec='None', label='GPR variance $\sigma$')[0]
# =============================================================================
    from matplotlib.patches import Circle
    circles=[]
    err = np.sqrt(np.diag(s))
    for i in range(len(s)):
        circle = Circle(m[i,:], err[i], color='lightblue', alpha=1)
        circles.append(circle)
        #label='GPR variance $\sigma$'
        ax.add_artist(circle)
    ax.set_xlabel(u"Longitudinal Distance from Start in $m$")
    ax.set_ylabel(u"Latitudinal Distance from Start in $m$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(circles[0])
    labels.append(u"95% confidence interval")
    ax.legend(handles, labels)
    if forTesting:
        def updateFusionInPositionSpace(val):
            t, m, s = asynchronous_GP(Data.Xacc, Data.Xgps, p_acc, p_gps,
                                   ssrbf.val, slrbf.val, ssdot.val, snoise_acc.val, snoise_gps.val)
            l_m.set_data(m[:,0], m[:,1])
    # =============================================================================
    #         l_sy.set_xy(np.array([np.concatenate([m[:,0], m[::-1,0]]),
    #                               np.concatenate([m[:,1] - 2 * np.sqrt(np.diag(s)),
    #                                              (m[:,1] + 2 * np.sqrt(np.diag(s)))[::-1]]
    #                               )
    #                     ]).T
    #         )
    # =============================================================================
    # =============================================================================
    #         global l_sy
    #         l_sy.remove()
    #         l_sy = ax.fill_between(m[:,0], m[:,1] - np.sqrt(np.diag(s)), m[:,1] + np.sqrt(np.diag(s)),
    #                            alpha=.3, color='b', edgecolor='None', label='GPR variance $\sigma$')
    # =============================================================================
    # =============================================================================
    #         l_sx.set_xy(np.array([np.concatenate([m[:,0] - np.sqrt(np.diag(s)),
    #                                              (m[:,0] + np.sqrt(np.diag(s)))[::-1]]),
    #                               np.concatenate([m[:,1], m[:,1][::-1]])]).T)
    # =============================================================================
            err = np.sqrt(np.diag(s))
            for i in range(len(s)):
                circles[i].center = m[i,:]
                circles[i].set_radius(err[i])
            fig.canvas.draw_idle()
        ssrbf.on_changed(updateFusionInPositionSpace)
        ssdot.on_changed(updateFusionInPositionSpace)
        slrbf.on_changed(updateFusionInPositionSpace)
        snoise_acc.on_changed(updateFusionInPositionSpace)
        snoise_gps.on_changed(updateFusionInPositionSpace)
    fig.show()
    
    #---HIER STEHT QUARK
    #QUARK
    
    
    #---8th plot will einfach nicht funktionieren -.-
    from scipy.interpolate import interp1d
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from matplotlib import colors
    u=np.array(m[:,0])
    r=np.array(1.96 * np.sqrt(np.diag(s)))
    f=interp1d(r,u)
    # walk along the circle
    p = np.linspace(0,2*np.pi,50)
    R,P = np.meshgrid(r,p)
    # transform them to cartesian system
    X,Y = R*np.cos(P),R*np.sin(P)
    Z=f(R)
# =============================================================================
#     norm = (r - r.min()) / (r.max() - r.min())
#     fcolors = cm.jet(norm)
# =============================================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Data.Xacc, p_acc[:,0], p_acc[:,1], label='double integrated acceleration')
    ax.plot(Data.Xgps, p_gps[:,0], p_gps[:,1], label='GPS position')
    ax.plot_surface(Z, X+m[:,0], Y+m[:,1], rstride=1, cstride=1,
                    linewidth=0, #facecolors=fcolors)
                    label="95% confidence interval of estimate")
    #ax.legend()
    fig.show()
    
    #---9th plot *sigh* Heteroscedastic Noise (just plotting each x and y seperately)
    #n=10000
    from SensorFusionGP import initialValues
    from Integration import rotatingIntegration
    #prepare position data
    p_gps = Data.latlonToMeter(Data.Ygps)[:,::-1]
    #prepare acceleration data
    v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, p_gps[:,0], p_gps[:,1])
    x0 = [np.interp(Data.Xacc[0], Data.Xgps[:2,0], p_gps[:2,0]),
          np.interp(Data.Xacc[0], Data.Xgps[:2,0], p_gps[:2,1])]
    #[22.13850709, -2.16845578, -0.42172239,  0.53603343, 23.83197455, -6.74910446]
    p_acc_x, p_acc_y = rotatingIntegration(Data.Xacc, Data.Yacc, x0[0], x0[1], v0[1], v0[0], forward[::-1])
    p_acc = np.array([p_acc_x, p_acc_y]).T
    #set hyperparameters
    s_rbf     = 20  #20  (0) only linear kernel -> reduces to linear regression
    l_rbf     = 2   #2   (0.3) jumping between the measurements
    s_dot     = 1   #1   (0) only rbf kernel -> prior mean is 0, where our estimate is drawn to
    noise_acc = 1   #1   (0.1) since we have many more acc measurements it dominates (note that ther is almost no uncertainty, even when the result is obviously bad)
    noise_gps = 50  #4   (50) this has a similar effect than the above but increased uncertainty
    forTesting = True
    noise_acc_cum = integrate.cumtrapz(
            integrate.cumtrapz(
                    np.repeat(noise_acc, len(Data.Xacc)),
                    Data.Xacc.flatten(),
                    initial=0.),
            Data.Xacc.flatten(),
            initial=0.
    )
    #GPR
    t, m, s = asynchronous_GP(Data.Xacc, Data.Xgps, p_acc, p_gps,
                              s_rbf, l_rbf, s_dot, noise_acc_cum, noise_gps)
    error = 1.96*np.sqrt(np.diag(s))
    #plotting
    fig = pyplot.figure(figsize=(10,4))
    fig.suptitle("Heteroscedastic GP Fusion of GPS and double integrated accelerometer data with cumulative noise\n"
       "($\sigma_{{rbf}}={0}, l_{{rbf}}={1}, \sigma_{{dot}}={2}, noise_{{acc}}={3}, noise_{{gps}}={4}$)".format(s_rbf, l_rbf, s_dot, noise_acc, noise_gps))
    ax = fig.add_subplot(121)
    ax.set_title("Longitudinal component")
    #ax.plot(Data.Xacc[0], x0[0], marker="x", color="red", label="initial position")
    ax.plot(Data.Xgps, p_gps[:,0], ".", label="GPS position")
    ax.plot(Data.Xacc, p_acc[:,0], "--", label="Integrated acceleration")
    ax.plot(t, m[:,0], label="Estimate")
    ax.fill_between(t[:,0], m[:,0]-error, m[:,0]+error, alpha=0.3,
                    label="95% confidence interval")
    ax.set_xlabel("Time in $s$")
    ax.set_ylabel("Longitudinal distance from start in $m$")
    ax.legend()
    ax = fig.add_subplot(122)
    ax.set_title("Latitudinal component")
    #ax.plot(Data.Xacc[0], x0[1], marker="x", color="red", label="initial position")
    ax.plot(Data.Xgps, p_gps[:,1], ".", label="GPS position")
    ax.plot(Data.Xacc, p_acc[:,1], "--", label="Integrated acceleration")
    ax.plot(t, m[:,1], label="Estimate")
    ax.fill_between(t[:,0], m[:,1]-error, m[:,1]+error, alpha=0.3,
                    label="95% confidence interval")
    ax.set_xlabel("Time in $s$")
    ax.set_ylabel("Latitudinal distance from start in $m$")
    ax.legend()    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.show()
    
    
    #---10th plot homoscedastic approach
# =============================================================================
#     rbf = RBF() * ConstantKernel()
#     dot = ConstantKernel() * DotProduct()
#     k = WhiteKernel(4., (1e-4, 1e3)) + rbf + dot
#     gpr = GaussianProcessRegressor(k, n_restarts_optimizer=32)
#     X_pred = np.concatenate((Data.Xgps, Data.Xacc))
#     sort = X_pred[:,0].argsort()
#     x_star = np.atleast_2d(np.linspace(np.min(X_pred), np.max(X_pred), n)).T
#     x = X_pred[sort]#[unique]
#     f = np.concatenate((p_gps, p_acc))[sort]#[unique]
#     gpr.fit(x, f)
#     m, s = gpr.predict(x_star, True)
#     t = x_star
#     error = 1.96*s
#     parameter = gpr.kernel_
#     fig = pyplot.figure(figsize=(10,4))
#     fig.suptitle("GP Fusion of GPS and double integrated accelerometer data \n"
#        "("+str(parameter)+")" )
#     ax = fig.add_subplot(121)
#     ax.set_title("Longitudinal component")
#     ax.plot(Data.Xgps, p_gps[:,0], ".", label="GPS position")
#     ax.plot(Data.Xacc, p_acc[:,0], "--", label="Integrated acceleration")
#     ax.plot(t, m[:,0], label="Estimate")
#     ax.fill_between(t[:,0], m[:,0]-error, m[:,0]+error, alpha=0.3,
#                     label="95% confidence interval")
#     ax.set_xlabel("Time in $s$")
#     ax.set_ylabel("Longitudinal distance from start in $m$")
#     ax.legend()
#     ax = fig.add_subplot(122)
#     ax.set_title("Latitudinal component")
#     ax.plot(Data.Xgps, p_gps[:,1], ".", label="GPS position")
#     ax.plot(Data.Xacc, p_acc[:,1], "--", label="Integrated acceleration")
#     ax.plot(t, m[:,1], label="Estimate")
#     ax.fill_between(t[:,0], m[:,1]-error, m[:,1]+error, alpha=0.3,
#                     label="95% confidence interval")
#     ax.set_xlabel("Time in $s$")
#     ax.set_ylabel("Latitudinal distance from start in $m$")
#     ax.legend()    
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.8)
#     fig.show()
# =============================================================================
    
    
    #---11th plot leastSqFitted Acceleration with Homoscedastic GPR
# =============================================================================
#     #[22.13850709, -2.16845578, -0.42172239,  0.53603343, 23.83197455, -6.74910446]
#     #set up acceleration according to the least square fit
#     calibratedYacc = Data.Yacc + [0, -0.42172238739533024, 0.5360334305256229]
#     from Integration import accToPos
#     p_acc_x, p_acc_y = accToPos(Data.Xacc, calibratedYacc,
#                                 18.97244945, -5.11213289, #start position
#                                 22.13850709, -2.16845578, #start speed
#                                 np.array([23.83197455, -6.74910446]))[0:2]
#     p_acc = np.array([p_acc_x, p_acc_y]).T
#     #fitting homoscedastic GP
#     #---------same like  plot 9 (sorry no time to make a function)-------------
#     rbf2 = RBF() * ConstantKernel()
#     dot2 = ConstantKernel() * DotProduct()
#     k2 = WhiteKernel(4., (1e-4, 1e4)) + rbf2 + dot2
#     gpr2 = GaussianProcessRegressor(k2, n_restarts_optimizer=32)
#     X_pred = np.concatenate((Data.Xgps, Data.Xacc))
#     sort = X_pred[:,0].argsort()
#     x = X_pred[sort]#[unique]
#     f = np.concatenate((p_gps, p_acc))[sort]#[unique]
#     gpr2.fit(x, f)
#     t = np.atleast_2d(np.linspace(np.min(x), np.max(x), n)).T
#     m, s = gpr2.predict(t, True)
#     error = 1.96*s
#     parameter = gpr2.kernel_
#     #plot it
#     fig = pyplot.figure(figsize=(10,4))
#     fig.suptitle("GP Fusion of GPS and least square fitted double integrated accelerometer data \n"
#        "("+str(parameter)+")" )
#     ax = fig.add_subplot(121)
#     ax.set_title("Longitudinal component")
#     ax.plot(Data.Xgps, p_gps[:,0], ".", label="GPS position")
#     ax.plot(Data.Xacc, p_acc[:,0], "--", label="Integrated acceleration")
#     ax.plot(t, m[:,0], label="Estimate")
#     ax.fill_between(t[:,0], m[:,0]-error, m[:,0]+error, alpha=0.3,
#                     label="95% confidence interval")
#     ax.set_xlabel("Time in $s$")
#     ax.set_ylabel("Longitudinal distance from start in $m$")
#     ax.legend()
#     ax = fig.add_subplot(122)
#     ax.set_title("Latitudinal component")
#     ax.plot(Data.Xgps, p_gps[:,1], ".", label="GPS position")
#     ax.plot(Data.Xacc, p_acc[:,1], "--", label="Integrated acceleration")
#     ax.plot(t, m[:,1], label="Estimate")
#     ax.fill_between(t[:,0], m[:,1]-error, m[:,1]+error, alpha=0.3,
#                     label="95% confidence interval")
#     ax.set_xlabel("Time in $s$")
#     ax.set_ylabel("Latitudinal distance from start in $m$")
#     ax.legend()    
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.8)
#     fig.show()
# =============================================================================
        
    #---plot 12+ (kernels)
    def kernel_plots():
        resolution = 100
        dt = np.atleast_2d(np.linspace(-5,5, resolution+1)).T
        #s, l, s2
        params = (np.ones((4, 4)) + np.diag([2]*4))[:,1:]
        for param in params:
            k = kernel(np.array([[0]]), dt, *param)
            fig = pyplot.figure()
            fig.suptitle(param)
            ax = fig.add_subplot(121)
            k = kernel(np.array([[0]]), dt, *param)
            ax.plot(dt[:,0], k[0,:], label="kernel function $K(0,x)")
            ax.set_ylim(-2, 30)
            ax.legend()
            ax = fig.add_subplot(122)
            k = kernel(np.array([[1]]), dt, *param)
            ax.plot(dt[:,0], k[0,:], label="kernel function $K(1,x)")
            ax.set_ylim(-2, 30)
            ax.legend()
        fig = pyplot.figure()
        fig.suptitle(str(param)+" noise=5")
        ax = fig.add_subplot(121)
        k = kernel(np.array([[0]]), dt, *param)
        cov00 = np.copy(k[:,resolution >> 1])
        k[:,resolution >> 1] = None
        ax.plot(dt[:,0], k[0,:], label="kernel function $K(0,x)")
        ax.plot(0, cov00 + 5, "_", label="nugget")
        ax.set_ylim(-2, 30)
        ax.legend()
        ax = fig.add_subplot(122)
        k = kernel(np.array([[0]]), dt, *param)
        cov00 = np.copy(k[:,resolution >> 1])
        k[:,resolution >> 1] = None
        ax.plot(dt[:,0], k[0,:], label="kernel function $K(1,x)")
        ax.plot(0, cov00 + 5, "_", label="nugget")
        ax.set_ylim(-2, 30)
        ax.legend()
    #kernel_plots()
    
"""
aufloesungGauss = 5
l = n**(0.5)*4
s = 10
noise = 0
x = np.atleast_2d(np.linspace(0, 100, n)).T #x vector
y = np.atleast_2d(np.linspace(-10, 10, aufloesungGauss)).T #y vector
#repeat x and y
#x,y = np.meshgrid(x, y) #repeat x and f n times in row/column direction
#m = np.atleast_2d(np.mean(y, axis=0)) #mean vector
m = 0
#get a Kernel
pairwise_dists = squareform(pdist(np.atleast_2d(np.linspace(0, 100, n)).T, 'sqeuclidean'))
kronecker_delta = np.zeros((n,n))
i = np.arange(n)
kronecker_delta[i,i] = 1
K = s**2 * np.exp(-pairwise_dists / l ** 2) + noise * kronecker_delta
plot = pyplot.figure()
#Gaussian
#f = np.concatenate((x, y), axis=1)
f = np.atleast_2d(np.random.normal(0, s, n)).T
G = (2*np.pi)**(-n/2) * np.linalg.det(K)**(-1/2) * np.exp(-1/2*(f-m).T @ np.linalg.inv(K) @ (f-m))
p = plot.add_subplot(111)
p.plot(x, G, "gray")
#plot acceleration
p = pyplot.figure()
ax = p.add_subplot(111, projection='3d')
#ax.title("Multivariate Gaussian")
ax.plot_surface(X=x, Y=y, Z=G)
"""
