# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:05:28 2018

@author: Stephan

http://www.michaelkeutel.de/blog/rotation-matrices-vector-basis/
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from math import cos
from math import sqrt
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

def accToPos(Xacc, Yacc, vx_0=0, vy_0=0):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    vx[0] = vx_0
    vy[0] = vy_0
    forward = np.array([1,0])
    for i in range(len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        lefthand = np.array([-forward[1], forward[0]])
        a = rotate(Yacc[i,1:], lefthand)
        aNext = rotate(Yacc[i+1,1:], lefthand)
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[1]*dt*dt
        vx[i+1] = vx[i] + dt*(a[0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[1] + aNext[1])/2
        forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
    return x, y, vx, vy


"""test circle"""
if __name__ == "__main__":
    centripetal=-5.0
    Xacc = np.array(range(1000))/10.0
    Yacc = np.array([[i, 0.0, centripetal] for i in Xacc])
    fig, ax = plt.subplots()
    #plt.plot(Yacc[:,0], Yacc[:,1])
    x, y, vx, vy = accToPos(Xacc, Yacc, 1.0, 0.0)
    line, = plt.plot(x, y)
    test, = plt.plot(vx, vy)
    axis = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(axis, 'centripetal force', valmin=-10.0, valmax=10.0, valinit=centripetal, valfmt='%0.2f')
    def update(val):
        Yacc = np.array([[i, 0.0, val] for i in Xacc])
        x, y, vx, vy = accToPos(Xacc, Yacc, 1.0, 0.0)
        line.set_xdata(x)
        line.set_ydata(y)
        test.set_xdata(vx)
        test.set_ydata(vy)
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()


def plotIntegration(Yacc, fig=None, ax=None):
    #lat lon zu Meter
    Ygpsnorm = Data.latlonToMeter(Ygps)
    if fig==None:
        fig=plt.figure()
    if ax==None:
        ax = fig.add_subplot(111, projection='3d')
    """plot it"""
    ax.plot(Xgps, Ygpsnorm[:,0], Ygpsnorm[:,1], 'bx', label=u'GPS-Position')
    """add gps prediction using 3D-GPR
    x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
    kernel = RBF(1, (0.01, 10)) * ConstantKernel(1.0, (1, 100)) + WhiteKernel(noise_level=0.1 ** 2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    gp.fit(Xgps, Ygpsnorm)
    y_pred, sigma = gp.predict(x, return_std=True)
    #ax.plot(x, y_pred[:,0], y_pred[:,1], 'r-', label=u'Prediction')
    """
    
    """https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet"""
    x, y = accToPos(Xacc, Yacc, 0, 0)
    
    YaccIntegratedY = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,1], x=Xacc, initial=0), x=Xacc, initial=0)
    YaccIntegratedZ = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,2], x=Xacc, initial=0), x=Xacc, initial=0)
    ax.plot(Xacc, YaccIntegratedY, YaccIntegratedZ, 'r-', label=u'Integration of acceleration data (simple)')
    ax.plot(Xacc, x, y, 'g--', label=u'Integration of acceleration data with rotation')
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
    print(m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    #ax.legend()


#test
"""
+1 +-2
   /\
+2 +-1

-1 +-2
   \/
-2 +-1


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

#def accToPos()
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