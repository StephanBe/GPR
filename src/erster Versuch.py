# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:29:29 2017

@author: Stephan
"""
from math import cos
import gmplot
import numpy
import csv
import os
from datetime import datetime
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import proj3d

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

numpy.random.seed(5)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def read(filename):
    Y = []
    X = []
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            X.append(datetime.strptime(row[0]+' '+row[1], '%d.%m.%Y %H:%M:%S,%f'))
            Y.append([float(x.replace(',', '.')) for x in row[2:]])
    #X = numpy.asarray(X)
    #Normierung auf den Start und Konvertierung in Sekunden
    X = numpy.atleast_2d([(x-min(X)).total_seconds() for x in X]).T
    Y = numpy.asarray(Y)
    return [X, Y]

def latlonToMeter(latlon):
    #Mittlere Höhe und Breite für Krümmung
    l = numpy.mean(latlon[:,0])
    #Meter pro Breitengrad
    m_per_deg_lat = 111132.92
    m_per_deg_lat -= 559.82 * cos(2.0 * l)
    m_per_deg_lat += 1.175 * cos(4.0 * l)
    m_per_deg_lat -= 0.0023 * cos(6.0 * l)
    #Neter pro Längengrad
    m_per_deg_lon = 111412.84 * cos(l)
    m_per_deg_lon -= 93.5 * cos(3.0 * l)
    m_per_deg_lon += 0.118 * cos(5.0 * l)
    #Normieren und umrechnen in Meter
    norm = []
    for x in latlon:
        norm.append([(x[0]-latlon[0,0])*m_per_deg_lat,
                     (x[1]-latlon[0,1])*m_per_deg_lon])
    return numpy.asarray(norm)
        

"""read data"""
prevdir=os.getcwd()
os.chdir('data\\Erprobung\\Crashinszinierung Galileo\\1')

"""read acceleration"""
Xacc, Yacc = read('AccelData.txt')

"""read GPS"""
Xgps, Ygps = read('GPSData.txt')

os.chdir(prevdir)

"""plot acceleration"""
pyplot.subplot(111)
pyplot.plot(Xacc, Yacc[:,0], 'rx', label=u'$a_x$')
pyplot.title("Acceleration with 1D-GPR")

"""GP regression acceleration x"""
x = numpy.atleast_2d(numpy.linspace(-5, 35, 1000)).T
kernel = RBF(0.1, (0.01, 10))*ConstantKernel(1.0, (1, 100))
noise = 0.5

gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=3)
gp.fit(Xacc, Yacc[:,0]-numpy.mean(Yacc[:,0]))
y_pred, sigma = gp.predict(x, return_std=True)
y_pred += numpy.mean(Yacc[:,0])
pyplot.plot(x, y_pred, 'r-', label=u'Prediction $a_x$')
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval')

pyplot.plot(Xacc, Yacc[:,1], 'gx', label=u'$a_y$')

"""GP regression acceleration y"""
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=3)
gp.fit(Xacc, Yacc[:,1]-numpy.mean(Yacc[:,1]))
y_pred, sigma = gp.predict(x, return_std=True)
y_pred += numpy.mean(Yacc[:,1])
pyplot.plot(x, y_pred, 'g-', label=u'Prediction $a_y$')
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='g', ec='None', label='95% confidence interval')
    
pyplot.plot(Xacc, Yacc[:,2], 'bx', label=u'$a_z$')

"""GP regression acceleration z"""
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=3)
gp.fit(Xacc, Yacc[:,2]-numpy.mean(Yacc[:,2]))
y_pred, sigma = gp.predict(x, return_std=True)
y_pred += numpy.mean(Yacc[:,2])
pyplot.plot(x, y_pred, 'b-', label=u'Prediction $a_z$')
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

pyplot.xlabel('$t$')
pyplot.ylabel('$f(t)$')
pyplot.ylim(min(Yacc.ravel())-1, max(Yacc.ravel())+1)
#pyplot.xlim(19,21)
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.show()

"""plot Google Maps with GPS"""
gmap = gmplot.GoogleMapPlotter(
        sum(Ygps[:,0])/float(len(Ygps[:,0])), 
        sum(Ygps[:,1])/float(len(Ygps[:,1])), 20,
        'AIzaSyC2I6z5RX44ZDn5z1-PiVFoEIIEVp5scKI')
gmap.plot(Ygps[:,0], Ygps[:,1], 'cornflowerblue', edge_width=3)
gmap.draw("googlemapsplot.html")
with open("googlemapsplot.html") as f:
    s = f.read()
with open("googlemapsplot.html", 'w') as f:
    s = s.replace("MapTypeId.ROADMAP", "MapTypeId.SATELLITE")
    f.write(s)
            

"""plot GPS"""
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Position over time and acceleration-vectors")
#lat lon zu Meter
Ygpsnorm = latlonToMeter(Ygps)
ax.plot(Xgps, Ygpsnorm[:,0], Ygpsnorm[:,1], 'b-', label=u'Measured')

"""
#add arrows pointing in direction of acceleration
kernel = RBF(1, (0.01, 3)) * ConstantKernel(1.0, (0.1, 10)) + WhiteKernel(noise_level=1.5 ** 2)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
gp.fit(Xacc, Yacc)
y_pred, sigma = gp.predict(Xgps, return_std=True)
for i, x in enumerate(Xgps.flatten()):
    a = Arrow3D([x, x+y_pred[i,0]],
            [Ygpsnorm[i,0], Ygpsnorm[i,0]+y_pred[i,1]],
            [Ygpsnorm[i,1], Ygpsnorm[i,1]+y_pred[i,2]],
            mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
"""
#add smoothed gps data
x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
kernel = RBF(1, (0.01, 10)) * ConstantKernel(1.0, (1, 100)) + WhiteKernel(noise_level=15 ** 2)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
gp.fit(Xgps, Ygpsnorm)
y_pred, sigma = gp.predict(x, return_std=True)

ax.plot(x, y_pred[:,0], y_pred[:,1], 'r-', label=u'Prediction')
ax.set_xlabel('$t$ in $s$')
ax.set_ylabel('$x$ in $m$')
ax.set_zlabel('$y$ in $m$')
ax.legend()
fig.show()

"""3d GP acceleration"""
fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.set_title('Acceleration with 3D-GPR')
ax.plot(Yacc[:,0], Yacc[:,1], Yacc[:,2], 'b-', label=u'Acceleration')

x = numpy.atleast_2d(numpy.linspace(min(Xacc), max(Xacc), 1000)).T
kernel = RBF(1, (0.01, 3)) * ConstantKernel(1.0, (0.1, 10)) + WhiteKernel(noise_level=1.5 ** 2)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
gp.fit(Xacc, Yacc)
y_pred, sigma = gp.predict(x, return_std=True)

ax.plot(y_pred[:,0], y_pred[:,1], y_pred[:,2], 'r-', label=u'Prediction')
ax.set_xlabel('$a_x$ in $m/s^2$')
ax.set_ylabel('$a_y$ in $m/s^2$')
ax.set_zlabel('$a_z$ in $m/s^2$')
pyplot.legend()
fig.show()


"""pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval')
"""

"""
            TESTING AREA 
""""""

def f(x):
    return x * numpy.sin(x) + 40

X = numpy.atleast_2d([1,2,3,4,5,6,7,9,10,11,12,17,18,19,20]).T
Y = f(X).ravel()

x = numpy.atleast_2d(numpy.linspace(-5, 25, 1000)).T
y = f(x).ravel()

#fig = pyplot.figure()
#pyplot.plot(X.ravel(), Y.ravel())

kernel = ConstantKernel(1.0, (0.001, 1000)) * RBF(10, (0.01, 100))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X, Y)
y_pred, sigma = gp.predict(x, return_std=True)

fig = pyplot.figure()
pyplot.plot(x, y, 'r:', label=u'$f(x) = x\,\sin(x)$')
pyplot.plot(X, Y, 'r.', markersize=10, label=u'Observations')
pyplot.plot(x, y_pred, 'b-', label=u'Prediction')
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
pyplot.xlabel('$x$')
pyplot.ylabel('$f(x)$')
pyplot.xlim(-5, 25)
pyplot.legend(loc='lower center')


"""""" 2d variant """
"""
from mpl_toolkits.mplot3d import Axes3D

def f2d(x):
    y1 = x
    y2 = x * numpy.sin(x) + 40
    return numpy.atleast_2d([y1,y2]).T[0,:,:]

y2d = f2d(x)
Y2d = f2d(X)

#kernel = ConstantKernel(1.0, (0.001, 1000)) * RBF(10, (0.01, 100))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9,
                              normalize_y=False)
#        theta0=0.1, thetaL=.1, thetaU=0.01)
gp.fit(X, Y2d)
y_pred, sigma = gp.predict(x, return_std=True)

fig = pyplot.figure()
#ax = fig.gca(projection='3d')
pyplot.plot(y2d.T[0,:], y2d.T[1,:], 'r:', label=u'$f(x) = (x; x\,\sin(x))$')
pyplot.plot(Y2d.T[0,:].ravel(), Y2d.T[1,:].ravel(), 
        'r.', markersize=10, label=u'Observations')
pyplot.plot(y_pred.T[0,:], y_pred.T[1,:], 'b-', label=u'Prediction')
pyplot.legend(loc='lower left')


"""