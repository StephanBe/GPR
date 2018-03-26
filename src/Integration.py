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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from scipy import integrate
from matplotlib.widgets import Slider
from math import sqrt

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
#vorher: android x (Yacc[:,0] = nach unten beschleunigen)
DOWN = 0
Yacc[:,DOWN] = tmp[:,0] #danach: Yacc[:,0] = beschleunigung nach unten
#vorher: android z (Yacc[:,2] = nach hinten beschleunigen)
FORWARD = 1
Yacc[:,FORWARD] = -tmp[:,2] #danach: Yacc[:,1] = beschleunigung nach vorn
#vorher: android y (Yacc[:,1] = nach links beschleunigen)
LEFT = 2
Yacc[:,LEFT] = tmp[:,1] #danach: Yacc[:,2] = beschleunigung nach links

"""return true if v > 1 km/h or any speed given"""
def isMoving(deltaXPosition, deltaYPosition, deltaTime, fasterThankmh=1.0):
    x = deltaXPosition
    y = deltaYPosition
    t = deltaTime
    if hasattr(x, "__len__"):
        x = x[0]
    if hasattr(y, "__len__"):
        y = y[0]
    if hasattr(t, "__len__"):
        t = t[0]
    speed = float(fasterThankmh)
    return((x*x + y*y) / (t*t) > 0.077160*speed*speed)

#https://en.wikipedia.org/wiki/Verlet_integration
def verlet_integration(Xacc, Yacc, vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    vx[0] = vx_0
    vy[0] = vy_0
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    for i in range(1, len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = 2*x[i] - x[i-1] + a[0]*dt*dt
        y[i+1] = 2*y[i] - y[i-1] + a[1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
    return x, y, vx, vy

#https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
def velocity_verlet_integration(Xacc, Yacc, vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
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

#my integration
def my_integration(t, Yacc, vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx[0] = vx_0
    vy[0] = vy_0
    #first iteration
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    if isMoving(x[1]-x[0], y[1]-y[0], dt):
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
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward)
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
        
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt) \
                and isMoving(x[i]-x[i-1], y[i]-y[i-1], t[i]-t[i-1]):
            #zentrale differenz für eine bessere Richtungsschätzung
            tb = t[i]   - t[i-1]
            ta = t[i+1] - t[i] # = dt
            forward[0] = 1.0/(tb+ta)*(tb/ta*x[i+1] + (ta**2 - tb**2)/(ta*tb)*x[i] - ta/tb*x[i-1])
            forward[1] = 1.0/(tb+ta)*(tb/ta*y[i+1] + (ta**2 - tb**2)/(ta*tb)*y[i] - ta/tb*y[i-1])
            #korrigiere den Schritt mit der besseren Richtungsschätzung
            a = rotate(Yacc[i,1:], forward)
            x[i+1] = x[i] + vx[i]*dt + 1/2*a[0]*dt*dt
            y[i+1] = y[i] + vy[i]*dt + 1/2*a[1]*dt*dt
            #korrigiere die nächste Beschleunigung mit dem besseren Schritt
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
            aNext = rotate(Yacc[i+1,1:], forward)
            #Integration durch trapezoidal rule
            vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
            vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
        
        
    return x, y, vx, vy

def accToPos(Xacc, Yacc, vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    return my_integration(Xacc, Yacc, vx_0, vy_0, forward)

def plotIntegration(Yacc, fig=None, ax=None):
    #lat lon zu Meter
    Ygpsnorm = Data.latlonToMeter(Ygps)
    x = Xgps
    lat, lon = Ygpsnorm[:,0], Ygpsnorm[:,1]
    if fig==None:
        fig=plt.figure()
    if ax==None:
        ax = fig.add_subplot(111, projection='3d')
    """plot it"""
    ax.view_init(0, -179)
    ax.plot(Xgps, lon, lat, 'x', label=u'GPS-Position') #plotting ^ lat and > lon
    
    """add gps prediction using 3D-GPR"""
    x = np.atleast_2d(np.linspace(min(Xgps), max(Xgps), 10000)).T
    kernel = RBF(1, (0.01, 10)) * ConstantKernel(1.0, (1, 100)) + WhiteKernel(noise_level=0.1 ** 2) + DotProduct(1,(0.1,10))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    gp.fit(Xgps, Ygpsnorm)
    y_pred, sigma = gp.predict(x, return_std=True)
    lat = y_pred[:,0]
    lon = y_pred[:,1]
    ax.plot(x, lon, lat, 'y-', label=u'Prediction')
    
    """get initial values"""
    v0 = np.array([0.0, 0.0])
    moving = 0
    for i in range(1, len(x)):
        dx = lon[i] - lon[i-1]
        dy = lat[i] - lat[i-1]
        dt = x[i] - x[i-1]
        if isMoving(dx, dy, dt):
            moving = i
            print(dx)
            print(dy)
            print(dt)
            print(i)
            print(sqrt(dx*dx+dy*dy)/dt)
            forward = np.array([dx, dy])
            print(forward)
            break
    lonFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,FORWARD], x=Xacc, initial=v0[0]), x=Xacc, initial=0)
    latFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,LEFT], x=Xacc, initial=v0[1]), x=Xacc, initial=0)
    """----ab hier weiter nach x/y lat/lon vertauschungen suchen-----"""
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'r-', label=u'$\int\int a_{original}$d$t$ assuming world coordinates (likely wrong)')
    lonFromAcc, latFromAcc = verlet_integration(Xacc, Yacc, v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'y--', label=u'$\int\int a_{original}$d$t$ (verlet integration)')
    lonFromAcc, latFromAcc = velocity_verlet_integration(Xacc, Yacc, v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, '--', label=u'$\int\int a_{original}$d$t$ (velocity verlet integration)')
    lonFromAcc, latFromAcc = my_integration(Xacc, Yacc, v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'b--', label=u'$\int\int a_{original}$d$t$ (my integration)')
    ax.scatter(x[moving], lon[moving], lat[moving], c="r", label=u'point determining the direction')
    
    """GP"""
    t_pred = np.atleast_2d(np.linspace(0, 30, 1000)).T
    #pos_pred, pos_sigma = gpr(Xacc, np.column_stack((x,y)), t_pred)
    acc_pred, acc_sigma = gpr(Xacc.reshape(-1,1), Yacc, t_pred)
    lonFromAcc, latFromAcc = accToPos(t_pred, acc_pred, v0[0], v0[1], forward)[0:2]
    ax.plot(t_pred, lonFromAcc, latFromAcc, 'g-.', label=u'$\int\int a_{GPR}$d$t$ with object coordinates (my integration)')
    """"""
    ax.set_xlabel(u'time in $s$')
    ax.set_ylabel(u'latitude in $m$')
    ax.set_zlabel(u'longitude in $m$')
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
    plotIntegration(Yacc, fig, ax)
    fig.legend()
    fig.tight_layout()
    fig.show()
    
    """test coordinates"""
    def plotCoordinateShuffle():
        """
        +1 +-2
           `/
        +2 +-1
           ||
        -1 +-2
           |\
        -2 +-1
           ||
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