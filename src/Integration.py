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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy import integrate
from matplotlib.widgets import Slider

import Data
from vectorRotation import rotate

def gpr(x_train, y_train, x_pred):
    #WhiteKernel for noise estimation (alternatively set alpha in GaussianProcessRegressor()) 
    #ConstantKernel for signal variance
    #RBF for length-scale
    kernel = RBF(0.1, (0.01, 10))*ConstantKernel(1.0, (0.1, 100)) + WhiteKernel(0.1, (0.01,1))
    #noise = 0.1
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
    mean = np.mean(y_train, 0)
    gp.fit(x_train, y_train-mean)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    y_pred += mean
    return y_pred, sigma

"""data"""
Xacc = Data.Xacc.flatten()
Yacc = Data.Yacc
Xgps = Data.Xgps
Ygps = Data.Ygps
Xgyr = Data.Xgyr
Ygyr = Data.Ygyr

tmp = np.copy(Yacc)
#android x -> car z  (unten)
Yacc[:,0] = tmp[:,0] #beschleunigung nach unten
#android y -> car -y (links)
Yacc[:,2] = -tmp[:,1] #beschleunigung nach rechts
#android z -> car -x (hinten)
Yacc[:,1] = -tmp[:,2] #beschleunigung nach vorn

#https://en.wikipedia.org/wiki/Verlet_integration
def verlet_integration(Xacc, Yacc, vx_0=0, vy_0=0):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    vx[0] = vx_0
    vy[0] = vy_0
    forward = np.array([1.0,0.0])
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    for i in range(1, len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = 2*x[i] - x[i-1] + a[0]*dt*dt
        y[i+1] = 2*y[i] - y[i-1] + a[1]*dt*dt
        #vertraue nicht auf die Richtung basierend auf Distanzen kleiner als 10 cm
        epsilon = 0.01
        if abs(x[i+1]-x[i]) > epsilon and abs(y[i+1]-y[i]) > epsilon:
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
    return x, y, vx, vy

#https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
def velocity_verlet_integration(Xacc, Yacc, vx_0=0, vy_0=0):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    vx[0] = vx_0
    vy[0] = vy_0
    forward = np.array([1.0,0.0])
    for i in range(len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = x[i] + vx[i]*dt + 1.0/2.0*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1.0/2.0*a[1]*dt*dt
        #vertraue nicht auf die Richtung basierend auf Distanzen kleiner als 10 cm
        epsilon = 0.01
        if abs(x[i+1]-x[i]) > epsilon and abs(y[i+1]-y[i]) > epsilon:
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
    return x, y, vx, vy

#my integration
def my_integration(t, Yacc, vx_0=0, vy_0=0):
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx[0] = vx_0
    vy[0] = vy_0
    forward = np.array([1.0,0.0])
    #first iteration
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    #vertraue nicht auf die Richtung basierend auf Distanzen kleiner als 10 cm
    epsilon = 0.01
    if abs(x[1]-x[0]) > epsilon and abs(y[1]-y[0]) > epsilon:
        forward = np.array([x[1]-x[0], y[1]-y[0]])
    aNext = rotate(Yacc[1,1:], forward)
    #Integration durch trapezoidal rule
    vx[1] = vx[0] + dt*(a[0] + aNext[0])/2
    vy[1] = vy[0] + dt*(a[1] + aNext[1])/2
    for i in range(1, len(t)-1):
        dt = t[i+1]-t[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[1]*dt*dt
        #vertraue nicht auf die Richtung basierend auf Distanzen kleiner als 10 cm
        epsilon = 0.01
        if abs(x[i+1]-x[i]) > epsilon and abs(y[i+1]-y[i]) > epsilon:
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
        
        #zentrale differenz für eine bessere Richtungsschätzung
        tb = t[i]   - t[i-1]
        ta = t[i+1] - t[i]
        forward[0] = 1.0/(tb+ta)*(tb/ta*x[i+1] + (ta**2 - tb**2)/(ta*tb)*x[i] - ta/tb*x[i-1])
        forward[1] = 1.0/(tb+ta)*(tb/ta*y[i+1] + (ta**2 - tb**2)/(ta*tb)*y[i] - ta/tb*y[i-1])
        #korrigiere den Schritt mit der besseren Richtungsschätzung
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[1]*dt*dt
        #korrigiere die nächste Beschleunigung mit dem besseren Schritt
        if abs(x[i+1]-x[i]) > epsilon and abs(y[i+1]-y[i]) > epsilon:
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
        
        
    return x, y, vx, vy

def accToPos(Xacc, Yacc, vx_0=0, vy_0=0):
    return my_integration(Xacc, Yacc, vx_0, vy_0)

def plotIntegration(Yacc, fig=None, ax=None):
    #lat lon zu Meter
    Ygpsnorm = Data.latlonToMeter(Ygps)
    if fig==None:
        fig=plt.figure()
    if ax==None:
        ax = fig.add_subplot(111, projection='3d')
    """plot it"""
    ax.plot(Xgps, Ygpsnorm[:,0], Ygpsnorm[:,1], 'x', label=u'GPS-Position')
    """add gps prediction using 3D-GPR
    x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
    kernel = RBF(1, (0.01, 10)) * ConstantKernel(1.0, (1, 100)) + WhiteKernel(noise_level=0.1 ** 2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    gp.fit(Xgps, Ygpsnorm)
    y_pred, sigma = gp.predict(x, return_std=True)
    #ax.plot(x, y_pred[:,0], y_pred[:,1], 'r-', label=u'Prediction')
    """
    
    
    
    YaccIntegratedY = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,1], x=Xacc, initial=0), x=Xacc, initial=0)
    YaccIntegratedZ = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,2], x=Xacc, initial=0), x=Xacc, initial=0)
    ax.plot(Xacc, YaccIntegratedY, YaccIntegratedZ, '-', label=u'Integration of acceleration data (assuming acceleration is measured in world coordinates)')
    x, y, vx, vy = verlet_integration(Xacc, Yacc, 0, 0)
    ax.plot(Xacc, x, y, '--', label=u'Integration of acceleration data with rotation (verlet integration)')
    x, y, vx, vy = velocity_verlet_integration(Xacc, Yacc, 0, 0)
    ax.plot(Xacc, x, y, '--', label=u'Integration of acceleration data with rotation (velocity verlet integration)')
    x, y, vx, vy = my_integration(Xacc, Yacc, 0, 0)
    ax.plot(Xacc, x, y, '--', label=u'Integration of acceleration data with rotation (my integration)')
    
    """GP
    t_pred = np.atleast_2d(np.linspace(0, 30, 1000)).T
    #pos_pred, pos_sigma = gpr(Xacc, np.column_stack((x,y)), t_pred)
    acc_pred, acc_sigma = gpr(Xacc.reshape(-1,1), Yacc, t_pred)
    YaccIntegratedY = integrate.cumtrapz(integrate.cumtrapz(acc_pred[:,1], x=t_pred.flatten(), initial=0), x=t_pred.flatten(), initial=0)
    YaccIntegratedZ = integrate.cumtrapz(integrate.cumtrapz(acc_pred[:,2], x=t_pred.flatten(), initial=0), x=t_pred.flatten(), initial=0)
    ax.plot(t_pred, YaccIntegratedY, YaccIntegratedZ, 'b-.', label=u'Integration of GP-predicted acceleration')
    """
    ax.set_xlabel(u'time in $s$')
    ax.set_ylabel(u'x position in $m$')
    ax.set_zlabel(u'y position in $m$')
    m = max(max(Yacc.flatten()),
            max(Ygpsnorm.flatten()),
            max(YaccIntegratedY.flatten()),
            max(YaccIntegratedZ.flatten())) * 1.1
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    #ax.legend()


if __name__ == "__main__":
    
    """test circle"""
    centripetal=-0.2
    N = 0.01
    xCircle = np.array(range(int(100*10**N)))/float(10**N)
    yCircle = np.array([[i, 0.0, centripetal] for i in xCircle])
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    #plt.plot(yCircle[:,0], yCircle[:,1])
    x, y, vx, vy = velocity_verlet_integration(xCircle, yCircle, 1.0, 0.0)
    line1, = plt.plot(x, y, "--", label='position with "velocity verlet" integration')
    x, y, vx, vy = verlet_integration(xCircle, yCircle, 1.0, 0.0)
    line2, = plt.plot(x, y, "--", label='position with "verlet" integration')
    x, y, vx, vy = my_integration(xCircle, yCircle, 1.0, 0.0)
    line3, = plt.plot(x, y, "-", label='position with my integration')
    test, = plt.plot(vx, vy, label='velocity')
    axis1 = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    axis2 = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor='green', label='resolution')
    slider1 = Slider(axis1, 'centripetal force', valmin=-1.0, valmax=1.0, valinit=centripetal, valfmt='%0.3f')
    slider2 = Slider(axis2, 'resolution', valmin=0.01, valmax=2.0, valinit=N, valfmt='%0.5f')
    slider2.valtext.set_text(10**N)
    def update(val):
        centripetal = slider1.val
        N = slider2.val
        xCircle = np.array(range(int(100*10**N)))/float(10**N)
        slider2.valtext.set_text(10**N)
        yCircle = np.array([[i, 0.0, centripetal] for i in xCircle])
        x, y, vx, vy = velocity_verlet_integration(xCircle, yCircle, 1.0, 0.0)
        line1.set_xdata(x)
        line1.set_ydata(y)
        x, y, vx, vy = verlet_integration(xCircle, yCircle, 1.0, 0.0)
        line2.set_xdata(x)
        line2.set_ydata(y)
        x, y, vx, vy = my_integration(xCircle, yCircle, 1.0, 0.0)
        line3.set_xdata(x)
        line3.set_ydata(y)
        test.set_xdata(vx)
        test.set_ydata(vy)
        fig.canvas.draw_idle()
    fig.legend()
    slider1.on_changed(update)
    slider2.on_changed(update)
    plt.show()

    """test integration of real acceleration data vs GPS"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(0, -179)
    plotIntegration(Yacc, fig, ax)
    fig.legend()
    fig.show()
    
    """test coordinates"""
    def plotCoordinateShuffle():
        """
        +1 +-2
           /\
        +2 +-1
        
        -1 +-2
           \/
        -2 +-1
        """
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
        
    #plotCoordinateShuffle()
    
    
    
    

"""
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