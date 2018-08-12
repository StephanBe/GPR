#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:31:34 2018

@author: stephan
"""
import numpy as np
from matplotlib import pyplot as plt
from myGP import kernel
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons
from scipy.integrate import cumtrapz
import Data
from myGP import derivative_GP as derivative_GP_old

n=1000

def derivative_GP(X_pos, X_acc, Y_pos, Y_acc, s1, l, s2, noisePos = 4., noiseAcc = 0.2, Ka = None):
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
    test = np.atleast_2d(np.linspace(X_pred[sort][unique][0], X_pred[sort][unique][-1], n)).T
    training = X_pred[sort][unique]
    #Add 2 colums (latent variables) to match the derivative kernel Ka.
    dt = np.diff(training.flatten())
    #--- 
    median_dt = np.median(dt)
    #dt = np.r_[[median_dt], [median_dt], dt]
    #training = np.r_[[training[0]-2*median_dt], [training[0]-median_dt], training]
    #--- 
    dt = np.r_[dt, [median_dt, median_dt]]
    #--- 
    training = np.r_[training, [training[-1]+median_dt], [training[-1]+2.0*median_dt]]
    
    #Set the kernel which chooses the observed values from latent variables.
    #choose Kp such that Y_pos = Kp @ latentY(X_pred)
# =============================================================================
#     Kp = np.zeros((len(X_pos), len(X_pred)))
#     for i in range(Kp.shape[1]):
#         if sort[i] < len(X_pos):
#             Kp[sort[i], i] = 1.
#     Kp = Kp[:,unique]
#     #Add 2 colums (latent variables) to match the derivative kernel Ka.
#     Kp = np.c_[Kp, np.zeros((len(X_pos),2))]
# =============================================================================
    Kp = np.eye(len(X_pos))
    Kp = np.c_[Kp, np.zeros((len(X_pos), len(X_acc)))]
    Kp = Kp[:,sort][:,unique]
    #Kp = np.c_[np.zeros((len(X_pos),1)), Kp, np.zeros((len(X_pos),1))]
    #--- 
    Kp = np.c_[Kp, np.zeros((len(X_pos),2))]
    
    #choose Ka such that Y_acc = Ka @ latentY(X_pred)
    Ka = np.zeros((len(X_acc), len(training)))
    for i in range(Ka.shape[1]-2):
        j = sort[unique2][i] - len(X_pos)
        if j >= 0:
            dt1 = dt[i]
            dt2 = dt[i+1]
            Ka[j, i]   = 2./(dt1*(dt1+dt2))
            Ka[j, i+1] = -2./(dt2*dt1)
            Ka[j, i+2] = 2./(dt2*(dt1+dt2))
            #Ka[j, i]   = 1./(dt1**2)
            #Ka[j, i+1] = -1./(dt1**2)-1./(dt1*dt2)
            #Ka[j, i+2] = 1./(dt1*dt2)
        
    #M = [Y_pos Y_acc]
    M = np.concatenate((Y_pos, Y_acc))
    #K = [Kp Ka]
    K = np.concatenate((Kp, Ka))
    
    #now we have set M = K @ latentY and can try to find this latent Y
    S = kernel(training, training, s1, l, s2)
    S2 = kernel(test, test, s1, l, s2)
    S12 = kernel(test, training, s1, l, s2)
    S21 = kernel(training, test, s1, l, s2)
    noise = [noisePos**2]*len(X_pos)+[noiseAcc**2]*len(X_acc)
    #noise[0] = 0
    #noise[len(X_pos)] = 0
    mu2_1 = S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    S2_1 = S2 - S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ K @ S21
    #resuts from setting K2 to I [Shimin Feng 2014]
    #mu2_1 = I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T + np.diag(noise)) @ M
    #S2_1 = S2 - I @ S12 @ K.T @ np.linalg.inv(K @ S @ K.T) @ K @ S21 @ I.T
    #in the original equations [Murray-Smith 2005]
    #mu2_1 = K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ M1
    #S2_1 = S2 - K2 @ S12 @ K1.T @ np.linalg.inv(K1 @ S @ K1.T) @ K1 @ S21 @ K2
    return test, mu2_1, S2_1


if __name__ is "__main__":
    
    #X_pos = np.linspace(0, 5, 21).reshape(-1,1)**2/5
    #Y_pos = X_pos**3 - X_pos**2 + 1
    #X_acc = np.linspace(0, X_pos[-1,0], int(np.ceil(X_pos[-1,0]*10))).reshape(-1,1)**2 /5.
    #Y_acc = 6*X_acc - 2#np.array([3.]*len(X_acc)).reshape(-1,1)
    X_pos = Data.Xgps#[1:,:]
    Ygps = Data.latlonToMeter(Data.Ygps)#[1:,:]
    X_acc = Data.Xacc_corrected2
    #use fitted data
    from vectorRotation import rotatedAcceleration
# =============================================================================
#     v0 = np.array([22.14242341, -2.17809353])
#     p0 = np.array([18.97244865, -5.11213311])
#     forward = np.array([23.83197455, -6.74910446])
#     Yacc = Data.Yacc + [0, -0.42233391105525775, 0.5373903677236002]
# =============================================================================
    v0 = np.array([0.48145699, 1.50722511])
    p0 = np.array([2.42586054, 3.584679])
    forward = np.array([0.78791583, 2.29049936])
    Yacc = Data.Yacc + [0, -0.22275387529091126, 0.0113005181983536]
    Yacc = rotatedAcceleration(X_acc, Yacc, v0[0], v0[1], forward)
    Y_acc = Yacc[:,1:]
    Y_pos = Ygps[:,:1]
    s1, l, s2, noiseP, noiseA = 20., 2., 1., 4., 4.
    
    #---figure 1
    fig = plt.figure()
    fig.subplots_adjust(bottom=.32)
    height = 0.03 #0.03
    vspace = 0.04 #0.04
    axgstart = fig.add_axes([0.05,0.01,0.23,height*6])
    axsigma  = fig.add_axes([0.45,0.01+vspace*4,0.40,height])
    axlength = fig.add_axes([0.45,0.01+vspace*3,0.40,height])
    axsigma2 = fig.add_axes([0.45,0.01+vspace*2,0.40,height])
    axnoiseP = fig.add_axes([0.45,0.01+vspace*1,0.40,height])
    axnoiseA = fig.add_axes([0.45,0.01+vspace*0,0.40,height])
    ssigma1 = Slider(axsigma,  '$\sigma_{rbf}$', valmin=0.01, valmax=50, valinit=s1, valfmt='%0.0f')
    slength = Slider(axlength, 'length scale',   valmin=0.01, valmax=5.0, valinit=l, valfmt='%0.1f')
    ssigma2 = Slider(axsigma2, '$\sigma_{dot}$', valmin=0.01, valmax=10, valinit=s2, valfmt='%0.1f')
    snoiseP = Slider(axnoiseP, 'noise of GPS',   valmin=0.0, valmax=50, valinit=noiseP, valfmt='%0.2f')
    snoiseA = Slider(axnoiseA, 'noise of acc.',  valmin=0.0, valmax=50, valinit=noiseA, valfmt='%0.2f')
    #sgstart  = Slider(axgstart,  'with GPS-start?',valmin=0.0, valmax=1, valinit=0.5, valfmt='%1.0f')
    bstart = CheckButtons(axgstart, ['with GPS-start?'], [True])
    ax = fig.add_subplot(111)
    def update(val):
        #s = int(not bstart.lines[0][0].get_visible())
        s = int(not bstart.get_status()[0])
        ax.clear()
        ax.set_title("Linear Transformation GPR Fusion with a new integration matrix")
        #s = int(round(sgstart.val))
        dx2 = X_acc[1:,0]-X_acc[:-1,0]
        #cumsum1 = np.cumsum(np.cumsum(Y_acc[ :-1,0]*dx2)*dx2)
        #cumsum2 = np.cumsum(np.cumsum(Y_acc[1:  ,0]*dx2)*dx2)
        #ax.plot(X_acc[ :-1,0], cumsum1, label="cumsum dervative data [:-1,:]")
        #ax.plot(X_acc[1:  ,0], cumsum2, label="cumsum dervative data [1:,:]")
        ia = ax.plot(X_acc, cumtrapz(cumtrapz(Y_acc, X_acc, axis=0, initial=0)+v0[1],
                                X_acc, axis=0, initial=0)+p0[1],
                       "--", color='orange', label="Int. acc. (lsq-fitted)")
        #ax.plot(X_acc, X_acc**3 - X_acc**2, label="analytically double integrated")
        t, m, e = derivative_GP_old(X_pos[s:,:], X_acc, Y_pos[s:,:], Y_acc,
                                ssigma1.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val)
        do = ax.fill_between(t.flatten(), m.flatten()+1.96*np.diag(e), m.flatten()-np.diag(e),
                        label="95% confidence interval (old)", alpha=.2, facecolor='red', edgecolor='none')
        eo = ax.plot(t.flatten(), m.flatten(), "r-.", label="Estimte (old)")
        t, m, e = derivative_GP(X_pos[s:,:], X_acc, Y_pos[s:,:], Y_acc,
                                ssigma1.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val)
        dn = ax.fill_between(t.flatten(), m.flatten()+1.96*np.diag(e), m.flatten()-np.diag(e),
                        label="95% confidence interval (new)", alpha=.2, facecolor='blue', edgecolor='none')
        en = ax.plot(t.flatten(), m.flatten(), "g-", label="Estimte (new)")
        gp = ax.plot(X_pos, Y_pos, "b.", label="GPS position")
        ax.set_ylim((np.min(Y_pos)-25, np.max(Y_pos)+15))#cumsum1)+20))
        h = [eo[0], en[0], gp[0], do, dn, ia[0]]
        ax.legend(h,[H.get_label() for H in h], ncol=2)
        ax.set_xlabel(u"Time in $s$")
        ax.set_ylabel(u"Latitudinal distance from start in $m$")
        
        fig.canvas.draw()
        #fig.canvas.flush_events()
    #oldval = 0
    #def updateStart(val):
    #    global oldval
    #    if oldval != round(sgstart.val):
    #        oldval = round(sgstart.val)
    #        update(val)
    
    ssigma1.on_changed(update)
    slength.on_changed(update)
    ssigma2.on_changed(update)
    snoiseA.on_changed(update)
    snoiseP.on_changed(update)
    bstart.on_clicked(update)
    update(0)
    fig.savefig("test.svg")
    #sgstart.on_changed(updateStart)
    
    def tri(x):
        dt = x[1:,:] - x[:-1,:]
        integr = np.tri(len(x))*np.insert(dt, 0, 1)
        return integr
    
    def integrate_tri(t, a, pos0, vel0):
        #x = np.insert(x, 0, [[1],[1]])
        #integr = tri(x)
        return np.tri(len(t))*np.insert(np.diff(t[:,0]), 0, 1) @ np.insert(
              (np.tri(len(t))*np.insert(np.diff(t[:,0]), 0, 1) @ np.insert(a, 0, [[pos0]], axis=0)),
                                       0, [[vel0]], axis=0)
    
    def differentiate_tri(x, y):
        integr = tri(x)
        return np.linalg.inv(integr) @ np.linalg.inv(integr) @ y
    
    def trapz(x):
        """double trapezoidal rule as matrix calculus works very good, but inverse oscilates"""
        dt = x[1:,:] - x[:-1,:]
        integr = np.tri(len(x), len(x), -1)*np.append(dt, 0)/2 +\
                 np.tri(len(x))*np.insert(dt, 0, 0)/2
        integr[0,0] = 1e-50
        return integr
    
    def integrate_trapz(x, y):
        """double trapezoidal rule as matrix calculus works very good, but inverse oscilates"""
        integr = trapz(x)
        return integr @ integr @ y
    
    def differentiate_trapz(x, y):
        """inverted double trapezoidal rule as matrix calculus oscilates"""
        dt = x[1:,:] - x[:-1,:]
        integr = np.tri(len(x), len(x), -1)*np.append(dt, 0)/2 +\
                 np.tri(len(x))*np.insert(dt, 0, 0)/2
        integr[0,0] = 0.0077
        diff = np.linalg.inv(integr)
        return diff @ diff @ y
    
    def differentiate_complicated(x, y):
        r"""
        Two times the central difference:
            
        .. math::
            f^{(2)}(x_i) = 
                \frac{2 f^{(1)}(\frac{x_i+x_{i+1}}{2}) - 2 f^{(1)}(\frac{x_{i-1}+x_i}{2})}{
                      x_{i+1}-x_{i-1}
                }
        
        with
        
        .. math::
            f^{(1)}(\frac{x_i+x_{i+1}}{2}) = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1}-x_i}
        
        https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/"""
        dt = x[1:,0] - x[:-1,0]
        diff = np.eye(len(x),len(x), -1) *\
               np.r_[2/(dt[:-1]*(dt[:-1]+dt[1:])), [0,0]] +\
               np.eye(len(x),len(x)) *\
               np.r_[[0], -2/(dt[1:]*dt[:-1]), [0]] +\
               np.eye(len(x),len(x), 1) *\
               np.r_[[0,0], 2/(dt[1:]*(dt[:-1]+dt[1:]))]
        diff[ 0, 0:3] = [1/(dt[ 0]**2), -1/(dt[ 0]**2)-1/(dt[ 1]*dt[ 0]), 1/(dt[ 1]*dt[ 0])]
        diff[-1,-3: ] = [1/(dt[-2]**2), -1/(dt[-2]**2)-1/(dt[-1]*dt[-2]), 1/(dt[-1]*dt[-2])]
        #diff[ 0, 0:2] = [-1/dt[ 0],1/dt[ 0]]
        #diff[-1,-2: ] = [-1/dt[-1],1/dt[-1]]
        #diff[0,0:3]=[1e-50,0,0]
        return diff @ y
    def integrate_complicated(x, y):
        """https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/"""
        dt = x[1:,0] - x[:-1,0]
        diff = np.eye(len(x),len(x), -1) *\
               np.r_[2/(dt[:-1]*(dt[:-1]+dt[1:])), [0,0]] +\
               np.eye(len(x),len(x)) *\
               np.r_[[0], -2/(dt[1:]*dt[:-1]), [0]] +\
               np.eye(len(x),len(x), 1) *\
               np.r_[[0,0], 2/(dt[1:]*(dt[:-1]+dt[1:]))]
        #diff[ 0, 0:3] = [1/(dt[ 0]**2), -1/(dt[ 0]**2)-1/(dt[ 1]*dt[ 0]), 1/(dt[ 1]*dt[ 0])]
        #diff[-1,-3: ] = [1/(dt[-2]**2), -1/(dt[-2]**2)-1/(dt[-1]*dt[-2]), 1/(dt[-1]*dt[-2])]
        
        #diff[ 0, 0:2] = [-1/dt[ 0],1/dt[ 0]]
        #diff[-1,-2: ] = [-1/dt[-1],1/dt[-1]]
        #diff[ 0,  :2] = [-2/(dt[0]*dt[1]), 2/(dt[1]*(dt[0]+dt[1]))]
        #diff[-1,-2: ] = [2/(dt[-2]*(dt[-2]+dt[-1])), -2/(dt[-2]*dt[-1])]
        diff[ 0, 0] = 1e50
        diff[-1,-1] = 1
        return np.linalg.inv(diff) @ y
    def centralDifference(x, y):
        dt = x[1:,:] - x[:-1,:]
        diff = np.diag(1/(2*dt),-1) + np.diag(1/(2*dt), 1)
        diff = diff / (2*dt)
        return diff @ y
    
    def differentiate(x, y):
        """backward difference"""
        dt = x[1:,:] - x[:-1,:]
        diff = -np.eye(len(x), len(x),-1) + np.eye(len(x))
        if x[0] == 0:
            diff = diff / np.r_[[[1]], dt]
        else:
            diff = diff / np.r_[[x[0]], dt]
        ddiff = diff @ diff
        return ddiff @ y
    def double_differentiate(x, y):
        dt = np.zeros(x.shape, np.float)
        dt[0:-1] = np.diff(x, axis=0)
        dt[-1] = dt[-2]
        dd1 = -np.eye(len(x), k=-1) + np.eye(len(x))
        dd2 = -np.eye(len(x)) + np.eye(len(x), k=1)
        dd1 = dd1/dt
        dd2 = dd2/dt
        dd = (dd2-dd1)*2/(dt+np.append(dt[1:,:], dt[-1:], axis=0))
        return dd @ y
    #---figure 2
    plt.figure()
    plt.title("Testing different integration and differentiation techniques")
    plt.plot(X_pos, Y_pos, "x", label="function (ground truth)")
    plt.plot(X_acc, Y_acc, "x", label="2nd derivative (ground truth)")
    #plt.plot(X_acc, X_acc**3 - X_acc**2, "x", label="integrated 2nd derivative (ground truth)")
    plt.plot(X_acc, integrate_trapz(X_acc, Y_acc),
             label="integration of 2nd derivative (trapz)")
    #plt.plot(X_pos, differentiate_trapz(X_pos, Y_pos),
    #         label="2nd derivative of function (trapz)")
    plt.plot(X_acc, integrate_complicated(X_acc, Y_acc),
             label="integration of 2nd derivative (complicated)")
    plt.plot(X_acc, cumtrapz(
                        cumtrapz(Y_acc, X_acc, initial=0, axis=0),
                        X_acc, initial=0, axis=0),
            "--", label="cumtrapz of 2nd derivative")
    plt.plot(X_pos, differentiate(X_pos, Y_pos),
             label="2nd derivative of function")
    plt.plot(X_pos, differentiate_complicated(X_pos, Y_pos),
             label="2nd derivative of function (complicated)")
    #plt.plot(X_pos, double_differentiate(X_pos, Y_pos),
    #         label="2nd derivative of function (double diff)")
    plt.legend()
    plt.show()
    
    
    
    
    #import Data
    #X_acc = Data.Xacc
    
    #---figure 3
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.21)
    dt = X_acc[1:,:] - X_acc[:-1,:]
    integr = np.tri(len(X_acc), len(X_acc), -1)*np.append(dt, 0)/2. +\
             np.tri(len(X_acc))*np.insert(dt, 0, 0.)/2.
    integr[0,0] = 1e-20
    d = dt.flatten()
    # =============================================================================
    # diff = (np.diag(np.r_[2/(d[:-1]*(d[:-1]+d[1:])), [0]], -1) +
    #           np.diag(np.r_[[0], -2/(d[1:]*d[:-1]), [0]]) +
    #           np.diag(np.r_[[0],  2/(d[1:]*(d[:-1]+d[1:]))], 1))
    # diff[0,0] = 1.
    # diff[-1,-1] = 1.
    # =============================================================================
    diff = -np.eye(len(X_acc), len(X_acc),-1) + np.eye(len(X_acc))
    diff /= np.r_[[[1e-30]], dt]
    axInteg = fig.add_subplot(221)
    axInteg.set_title("Inverse double differentiation matrix")
    i = axInteg.imshow(integr @ integr)
    fig.colorbar(i)
    axDeriv = fig.add_subplot(222)
    axDeriv.set_title("Double differentiation matrix")
    d = axDeriv.imshow(np.linalg.inv(integr @ integr))
    fig.colorbar(d)
    axPlot  = fig.add_subplot(212)
    axPlot.set_title("Plot")
    axPlot.plot(X_acc, Y_acc, "--", label="acc")
    li, = axPlot.plot(X_acc, np.linalg.inv(diff @ diff) @ (Y_acc), label="integrated acc")
    diff = -np.eye(len(X_pos), len(X_pos),-1) + np.eye(len(X_pos))
    diff /= np.r_[[[1e-30]], np.diff(X_pos, axis=0)]
    ld, = axPlot.plot(X_pos, diff @ diff @ (Y_pos), label="differentiated pos")
    axPlot.plot(X_pos, Y_pos, "x", label="pos")
    #axPlot.plot(X_acc, 3*X_acc**2 - 2*X_acc, "x", label="velocity")
    axPlot.legend()
    #axPlot.set_ylim((-20,200))
    
    axsstart   = fig.add_axes([0.38,0.13,0.50,0.03])
    axsstart2  = fig.add_axes([0.38,0.09,0.50,0.03])
    axsappend  = fig.add_axes([0.38,0.05,0.50,0.03])
    axsappend2 = fig.add_axes([0.38,0.01,0.50,0.03])
    sstart   = Slider(axsstart,'start', valmin=-1, valmax=1, valinit=0.01, valfmt='%0.5f')
    sstart2  = Slider(axsstart2,'start2', valmin=-1e2, valmax=1e2, valinit=0.005, valfmt='%0.5f')
    sappend  = Slider(axsappend,'append', valmin=-25., valmax=2., valinit=0., valfmt='%0.5f')
    sappend2 = Slider(axsappend2,'append2', valmin=-2, valmax=25, valinit=0, valfmt='%0.5f')
    def update2(val):
    # =============================================================================
    #     #testing trapezodial integration (suffers from heavy oscilation)
    #     integr = np.tri(len(X_acc), len(X_acc), -1)*np.append(dt, sappend.val)/2. +\
    #              np.tri(len(X_acc))*np.insert(dt, 0, sstart2.val)/2.
    #     integr[0,0] = sstart.val
    #     i.set_data(integr @ integr)
    #     d.set_data(np.linalg.inv(integr) @ np.linalg.inv(integr))
    #     li.set_ydata(integr @ integr @ (6*X_acc - 2))
    #     ld.set_ydata(np.linalg.inv(integr @ integr) @ (X_acc**3 - X_acc**2 + 1))
    # =============================================================================
    # =============================================================================
    #     #testing forward difference
    #     diff = -np.eye(len(X_acc)) + np.eye(len(X_acc),len(X_acc),1)
    #     diff = diff / np.r_[dt,[[sstart.val]]]
    #     i.set_data(np.linalg.inv(diff @ diff))
    #     d.set_data(diff @ diff)
    #     li.set_ydata(np.linalg.inv(diff @ diff) @ (6*X_acc - 2))
    #     ld.set_ydata(diff @ diff @ (X_acc**3 - X_acc**2 + 1))
    # =============================================================================
    # =============================================================================
    #     #testing double central difference
    #     diff = (np.diag(np.append(2/(dt[ :-1]*(dt[:-1]+dt[1:])), sappend2.val), -1) +
    #             np.diag(np.append(np.append(sstart.val, -2/(dt[1:]* dt[:-1])), sappend.val)) +
    #             np.diag(np.append(sstart2.val, 2/(dt[1:]*(dt[:-1]+dt[1:]))),  1))
    #     #diff[ 0, 0] = 1
    #     #diff[-1,-3: ] = [1/(dt[-2]*dt[-1]), -1/(dt[-2]*dt[-1])-1/(dt[-1]**2), 1/(dt[-1]**2)]
    #     #diff[-1,-3:] = [1/dt[-2], -(1/dt[-2]+1/dt[-1]), 1/dt[-1]]/dt[-1]
    #     integr = np.linalg.inv(diff)
    #     i.set_data(integr)
    #     i.set_clim(integr.min(), integr.max())
    #     d.set_data(diff)
    #     d.set_clim(diff.min(), diff.max())
    #     li.set_ydata(np.linalg.inv(diff) @ (6*X_acc - 2))
    #     ld.set_ydata(diff @ (X_acc**3 - X_acc**2 + 1))
    # =============================================================================
        #testing back difference (only one working, but bad accuracy)
        inte = -np.eye(len(X_acc), len(X_acc),-1) + np.eye(len(X_acc))
        inte /= np.r_[[[sstart.val]], np.diff(X_acc, axis=0)]
        inte = np.linalg.inv(inte @ inte)
        li.set_ydata(inte @ (Y_acc))
        diff = -np.eye(len(X_pos), len(X_pos),-1) + np.eye(len(X_pos))
        diff /= np.r_[[[sstart2.val]], np.diff(X_pos, axis=0)]
        diff = diff @ diff
        ld.set_ydata(diff @ (Y_pos))
        #li.set_ydata(np.linalg.inv(diff @ diff) @ (6*X_acc - 2))
        #ld.set_ydata(diff @ diff @ (X_acc**3 - X_acc**2 + 1))
        i.set_data(inte)
        d.set_data(diff)
        #---estimate
        t, m, e = derivative_GP(X_pos, X_acc, Y_pos, Y_acc, ssigma1.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val, diff @ diff)
        axPlot.fill_between(t.flatten(), m.flatten()+np.diag(e), m.flatten()-np.diag(e),
                        label="std deviation (new)", color="yellow", alpha=0.5)
        axPlot.plot(t.flatten(), m.flatten(), "y", label="Estimte (new)")
        t, m, e = derivative_GP_old(X_pos, X_acc, Y_pos, Y_acc,
                                ssigma1.val, slength.val, ssigma2.val, snoiseP.val, snoiseA.val)
        axPlot.fill_between(t.flatten(), m.flatten()+np.diag(e), m.flatten()-np.diag(e),
                        label="std deviation (old)", color="lightgreen", alpha=0.5)
        axPlot.plot(t.flatten(), m.flatten(), "g", label="Estimte (old)")
        
        fig.canvas.draw_idle()
        
    sstart2.on_changed(update2)
    sstart.on_changed(update2)
    sappend.on_changed(update2)
    sappend2.on_changed(update2)
    fig.show()