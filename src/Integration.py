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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from scipy import integrate
from matplotlib.widgets import Slider
from math import sqrt, pi, atan2

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
#Yacc[:,LEFT] = np.zeros(Yacc[:,LEFT].shape)

"""return true if v > 1 km/h or any speed given"""
def isMoving(deltaXPosition, deltaYPosition, deltaTime, fasterThankmh=1.0):
    x = deltaXPosition
    y = deltaYPosition
    t = deltaTime
    if t*t == 0.:
        return False
    if hasattr(x, "__len__"):
        x = x[0]
    if hasattr(y, "__len__"):
        y = y[0]
    if hasattr(t, "__len__"):
        t = t[0]
    speed = float(fasterThankmh)
    return((x*x + y*y) / (t*t) > 0.077160*speed*speed)

# =============================================================================
# def get_rotation(time, acceleration, initial_velocity):
#     velocity = np.zeros(acceleration.shape)
#     velocity[0,:] = initial_velocity
#     for i in range(len(acceleration)-1):
#         f  = velocity[i,:]
#         a  = acceleration[i,:]
#         v  = velocity[i,:] 
#         dt = time[i+1] - time[i]
#         change = 1.
#         for it in range(1):
#             if change < 1e-6:
#                 break
#             before = f
#             R = np.array([[f[0], -f[1]],
#                           [f[1],  f[0]]]) * 1 / sqrt(f.T @ f)
#             f = R @ a * dt + v
#             change = np.sum(np.abs(before - f))
#             print(f)
#             #print("current change "+str(change))
#         velocity[i+1,:] = f
#     return velocity
# =============================================================================


#https://en.wikipedia.org/wiki/Verlet_integration
def verlet_integration(Xacc, Yacc, x0=0., y0=0., vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    x[0] = x0
    y[0] = y0
    vx[0] = vx_0
    vy[0] = vy_0
    dt = Xacc[1]-Xacc[0]
    a = rotate(Yacc[0,1:], forward)
    x[1] = x[0] + vx[0]*dt + 1.0/2.0*a[0]*dt*dt
    y[1] = y[0] + vy[0]*dt + 1.0/2.0*a[1]*dt*dt
    for i in range(1, len(Xacc)-1):
        dt = Xacc[i+1]-Xacc[i]
        a = rotate(Yacc[i,1:], forward)
        x[i+1] = 2*x[i] - x[i-1] + a[0]*dt*dt
        y[i+1] = 2*y[i] - y[i-1] + a[1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward = np.array([x[i+1]-x[i], y[i+1]-y[i]])
    return x, y, vx, vy

#https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
def velocity_verlet_integration(Xacc, Yacc, x0=0., y0=0., vx_0=0, vy_0=0, forward=np.array([1.0, 0.0])):
    vx = np.zeros(len(Xacc))
    vy = np.zeros(len(Xacc))
    x = np.zeros(len(Xacc))
    y = np.zeros(len(Xacc))
    x[0] = x0
    y[0] = y0
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
def my_integration(t, Yacc, x0=0., y0=0., vx_0=0., vy_0=0.,
                   forward=np.array([1.0, 0.0]), return_rotated_acceleration=False):
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = vx_0
    vy[0] = vy_0
    tmp = forward
    forward = np.zeros((len(t), 2))
    forward[0:,:] = tmp
    #first iteration
    dt = Xacc[1]-Xacc[0]
    a = np.zeros((len(t), 2))
    a[0,:] = rotate(Yacc[0,1:], forward[0,:])
    x[1] = x[0] + vx[0]*dt + 1.0/2.0*a[0,0]*dt*dt
    y[1] = y[0] + vy[0]*dt + 1.0/2.0*a[0,1]*dt*dt
    if isMoving(x[1]-x[0], y[1]-y[0], dt):
        forward[1:,:] = np.array([x[1]-x[0], y[1]-y[0]])
    aNext = rotate(Yacc[1,1:], forward[1,:])
    #Integration durch trapezoidal rule
    vx[1] = vx[0] + dt*(a[0,0] + aNext[0])/2
    vy[1] = vy[0] + dt*(a[0,1] + aNext[1])/2
    for i in range(1, len(t)-1):
        dt = t[i+1]-t[i]
        a[i,:] = rotate(Yacc[i,1:], forward[i,:])
        x[i+1] = x[i] + vx[i]*dt + 1/2*a[i,0]*dt*dt
        y[i+1] = y[i] + vy[i]*dt + 1/2*a[i,1]*dt*dt
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt):
            forward[(i+1):,:] = np.array([x[i+1]-x[i], y[i+1]-y[i]])
        aNext = rotate(Yacc[i+1,1:], forward[i+1,:])
        #Integration durch trapezoidal rule
        vx[i+1] = vx[i] + dt*(a[i,0] + aNext[0])/2
        vy[i+1] = vy[i] + dt*(a[i,1] + aNext[1])/2
        
        if isMoving(x[i+1]-x[i], y[i+1]-y[i], dt) \
                and isMoving(x[i]-x[i-1], y[i]-y[i-1], t[i]-t[i-1]):
            #zentrale differenz für eine bessere Richtungsschätzung
            tb = t[i]   - t[i-1]
            ta = t[i+1] - t[i] # = dt
            forward[i:,0] = 1.0/(tb+ta)*\
                (tb/ta*x[i+1] + (ta**2 - tb**2)/(ta*tb)*x[i] - ta/tb*x[i-1])
            forward[i:,1] = 1.0/(tb+ta)*\
                (tb/ta*y[i+1] + (ta**2 - tb**2)/(ta*tb)*y[i] - ta/tb*y[i-1])
            #korrigiere den Schritt mit der besseren Richtungsschätzung
            a[i,:] = rotate(Yacc[i,1:], forward[i,:])
            x[i+1] = x[i] + vx[i]*dt + 1/2*a[i,0]*dt*dt
            y[i+1] = y[i] + vy[i]*dt + 1/2*a[i,1]*dt*dt
            #korrigiere die nächste Beschleunigung mit dem besseren Schritt
            forward[(i+1):,:] = np.array([x[i+1]-x[i], y[i+1]-y[i]])
            aNext = rotate(Yacc[i+1,1:], forward[i+1,:])
            #Integration durch trapezoidal rule
            vx[i+1] = vx[i] + dt*(a[i,0] + aNext[0])/2
            vy[i+1] = vy[i] + dt*(a[i,1] + aNext[1])/2
        
    if return_rotated_acceleration:
        return x, y, vx, vy, forward, a
    else:
        return x, y, vx, vy, forward

def rotatedIntegration(t, a, x0=0., y0=0., vx_0=0., vy_0=0., forward=np.array([1.,0.]), return_velocity=False):
    x, y, vx, vy, forward, a = my_integration(t, a, 0., 0., 1., 0., forward, True)
    v0 = np.repeat([[vx_0, vy_0]], len(t), axis=0) #rotated starting velocity
# =============================================================================
#     for i in range(len(t)):
#         v0[i,:] = rotate(v0[i,:], forward[i,:])
# =============================================================================
    xri = integrate.cumtrapz(integrate.cumtrapz(a[:,0], t, initial=0.)+v0[:,0], t, initial=x0)
    yri = integrate.cumtrapz(integrate.cumtrapz(a[:,1], t, initial=0.)+v0[:,1], t, initial=y0)
# =============================================================================
#     xri = integrate.cumtrapz(integrate.cumtrapz(a[:,0], t, initial=vx_0), t, initial=x0)
#     yri = integrate.cumtrapz(integrate.cumtrapz(a[:,1], t, initial=vy_0), t, initial=y0)
# =============================================================================
    if return_velocity:
        return xri, yri, vx, vy
    else:
        return xri, yri
# =============================================================================
#     #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
#     xri = np.zeros(len(t))
#     yri = np.zeros(len(t))
#     v0 = np.repeat([[vx_0, vy_0]], len(t), axis=0) #rotated starting velocity
#     for i in range(len(t)):
#         #a = rotate(yCircle[i,1:], forward[i,:])
#         forw = np.array([vx[i], vy[i]])
#         rotated_a = rotate(a[i,1:], forw)
#         v0[i,:] = rotate(v0[i,:], forw)
#         #a = rotate(yCircle[i,1:], v[i,:])
#         xri[i] = rotated_a[0]
#         yri[i] = rotated_a[1]
#     xri = integrate.cumtrapz(integrate.cumtrapz(xri, t, initial=0.)+v0[:,0], t, initial=x0)
#     yri = integrate.cumtrapz(integrate.cumtrapz(yri, t, initial=0.)+v0[:,1], t, initial=y0)
#     if return_velocity:
#         return xri, yri, vx, vy
#     else:
#         return xri, yri
# =============================================================================

def accToPos(Xacc, Yacc, x0=0., y0=0., vx_0=0., vy_0=0., forward=np.array([1.0, 0.0])):
    return my_integration(Xacc, Yacc, x0, y0, vx_0, vy_0, forward)

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
    xGP = np.atleast_2d(np.linspace(min(Xgps), max(Xgps), 1000)).T
    kernel = RBF(1, (0.1, 100)) *\
             ConstantKernel(1.0, (1, 100)) +\
             DotProduct(0.01,(0.01,5)) *\
             ConstantKernel(1.0, (1, 100))# + WhiteKernel(noise_level=0.1 ** 2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    gp.fit(Xgps, Ygpsnorm)
    y_pred, sigma = gp.predict(xGP, return_std=True)
    latGP = y_pred[:,0]
    lonGP = y_pred[:,1]
    ax.plot(xGP, lonGP, latGP, 'y-', label=u'Prediction')
    samples = gp.sample_y(xGP, 20)
    for i in range(samples.shape[2]):
        latGP = samples[:,0,i]
        lonGP = samples[:,1,i]
        ax.plot(xGP, lonGP, latGP, 'y--', alpha=.5, linewidth=0.5)
    
    """get initial values"""
    v0 = None
    position0 = gp.predict(Xacc[0])[0,::-1]
    moving = 0
    for i in range(1, len(x)):
        dx = lon[i] - lon[i-1]
        dy = lat[i] - lat[i-1]
        dt = x[i] - x[i-1]
        if v0 is None and x[i] > Xacc[0]:
            v0 = np.array([dx/dt, dy/dt]).flatten()
            print("setting initial speed to"+str(v0))
        if x[i] > Xacc[0] and isMoving(dx, dy, dt):
            moving = i
            print("calculating initial forward direction...")
            print("dx: "+str(dx))
            print("dy: "+str(dy))
            print("dt: "+str(dt))
            print("first noticable movement at iteration: "+str(i))
            print("speed in m/s: "+str(sqrt(dx*dx+dy*dy)/dt))
            forward = np.array([dx, dy])
            print("calculated initial forward vector: "+str(forward))
            print("(vector normalization is done during rotation)")
            break
    lonFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,FORWARD], x=Xacc, initial=v0[0]), x=Xacc, initial=0)
    latFromAcc = integrate.cumtrapz(integrate.cumtrapz(Yacc[:,LEFT], x=Xacc, initial=v0[1]), x=Xacc, initial=0)
    """----ab hier weiter nach x/y lat/lon vertauschungen suchen-----"""
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'r-', label=u'$\int\int a_{original}$d$t$ assuming world coordinates (likely wrong)')
    lonFromAcc, latFromAcc = verlet_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'y--', label=u'$\int\int a_{original}$d$t$ (verlet integration)')
    lonFromAcc, latFromAcc = velocity_verlet_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, '--', label=u'$\int\int a_{original}$d$t$ (velocity verlet integration)')
    lonFromAcc, latFromAcc = my_integration(Xacc, Yacc, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
    ax.plot(Xacc, lonFromAcc, latFromAcc, 'b--', label=u'$\int\int a_{original}$d$t$ (my integration)')
    ax.scatter(x[moving], lon[moving], lat[moving], c="r", label=u'point determining the initial direction')
    
    """GP"""
    t_pred = np.atleast_2d(np.linspace(min(Xacc), max(Xacc), 10000)).T
    #pos_pred, pos_sigma = gpr(Xacc, np.column_stack((x,y)), t_pred)
    acc_pred, acc_sigma = gpr(Xacc.reshape(-1,1), Yacc, t_pred)
    #Normalization here, knowing the car does not move for the first 120
    #data entries is cheating:
    #acc_pred[:,1:3] = acc_pred[:,1:3]-np.mean(acc_pred[0:120,1:3], axis=0)
    lonFromAcc, latFromAcc = accToPos(t_pred, acc_pred, position0[0], position0[1], v0[0], v0[1], forward)[0:2]
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
    fig, ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(bottom=0.15)
    #plt.plot(yCircle[:,0], yCircle[:,1])
    xvvi, yvvi = velocity_verlet_integration(xCircle, yCircle, 0., 0., 1., 0.)[0:2]
    line1, = plt.plot(xvvi, yvvi, "--", label='position with "velocity verlet" integration')
    xvi, yvi = verlet_integration(xCircle, yCircle, 0., 0., 1., 0.)[0:2]
    line2, = plt.plot(xvi, yvi, "--", label='position with "verlet" integration')
    x, y, vx, vy, forward = my_integration(xCircle, yCircle, 0., 0., 1., 0.)
    line3, = plt.plot(x, y, "-", label='position with my integration')
    #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
    xri, yri = rotatedIntegration(xCircle, yCircle, 0., 0., 1., 0.)
    line4, = plt.plot(xri, yri, ".-", label='position with simple double integration of rotated a')
    test, = plt.plot(vx, vy, label='velocity')
    axis1 = plt.axes([0.28, 0.01, 0.58, 0.03], facecolor='lightgoldenrodyellow')
    axis2 = plt.axes([0.28, 0.05, 0.58, 0.03], facecolor='green', label='resolution')
    slider1 = Slider(axis1, 'centripetal force', valmin=-1.0, valmax=1.0, valinit=centripetal, valfmt='%0.3f')
    slider2 = Slider(axis2, 'resolution', valmin=0.01, valmax=2.0, valinit=N, valfmt='%0.5f')
    slider2.valtext.set_text(round((10**N)*100.0)/100.0)
    def update(val):
        centripetal = slider1.val
        N = slider2.val
        xCircle = np.array(range(int(100*10**N)))/float(10**N)
        slider2.valtext.set_text(round((10**N)*100.0)/100.0)
        yCircle = np.array([[i, 0.0, centripetal] for i in xCircle])
        xvvi, yvvi = velocity_verlet_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)[0:2]
        line1.set_xdata(xvvi)
        line1.set_ydata(yvvi)
        xvi, yvi = verlet_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)[0:2]
        line2.set_xdata(xvi)
        line2.set_ydata(yvi)
        x, y, vx, vy, forward = my_integration(xCircle, yCircle, 0., 0., 1.0, 0.0)
        line3.set_xdata(x)
        line3.set_ydata(y)
        test.set_xdata(vx)
        test.set_ydata(vy)
        #v = get_rotation(xCircle, yCircle[:,1:], np.array([1.,0.]))
        xri, yri = rotatedIntegration(xCircle, yCircle, 0., 0., 1., 0.)
        line4.set_xdata(xri)
        line4.set_ydata(yri)
        fig.canvas.draw_idle()
    fig.legend()
    slider1.on_changed(update)
    slider2.on_changed(update)
    plt.show()
    #plot angle over iteration
    plt.figure()
    plt.title("angle of the circular movement (from integrating accel.)")
    plt.plot([180 / pi * atan2(forward[i, 0], forward[i, 1])
        for i in range(len(forward))], label="from forward vector")
    plt.plot([180 / pi * atan2(vx[i], vy[i])
        for i in range(len(forward))], label="from velocity vector")
    plt.ylabel("angle in °")
    plt.legend()
    plt.show()
    
# =============================================================================
#     """test integration of real acceleration data vs GPS"""
#     fig = plt.figure(figsize=(7.,7.))
#     ax = fig.add_subplot(111, projection='3d')
#     plotIntegration(Yacc, fig, ax)
#     fig.legend(loc='lower left', fontsize='small')
#     #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1, 1, 1]))
#     fig.subplots_adjust(top=1.1, right=1.1)
#     #fig.tight_layout()
#     fig.show()
# =============================================================================
    
    
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