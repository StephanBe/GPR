# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 07:06:37 2018

@author: Stephan
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy
from mpl_toolkits.mplot3d import proj3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import Data

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
  

"""data"""
Xacc = Data.Xacc
Yacc = Data.Yacc
Xgps = Data.Xgps
Ygps = Data.Ygps
Xgyr = Data.Xgyr
Ygyr = Data.Ygyr

tmp = numpy.copy(Yacc)
#android x -> car z  (unten)
Yacc[:,0] = tmp[:,0] #beschleunigung nach unten
#android y -> car -y (links)
Yacc[:,2] = -tmp[:,1] #beschleunigung nach rechts
#android z -> car -x (hinten)
Yacc[:,1] = -tmp[:,2] #beschleunigung nach vorn



#lat lon zu Meter
Ygpsnorm = Data.latlonToMeter(Ygps)


"""plot it"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Xgps, Ygpsnorm[:,0], Ygpsnorm[:,1], 'b-', label=u'GPS-Position')

"""add gps prediction using 3D-GPR"""
x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
kernel = RBF(1, (0.01, 10)) * ConstantKernel(1.0, (1, 100)) + WhiteKernel(noise_level=15 ** 2)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
gp.fit(Xgps, Ygpsnorm)
y_pred, sigma = gp.predict(x, return_std=True)
ax.plot(x, y_pred[:,0], y_pred[:,1], 'r-', label=u'Prediction')

"""add arrows pointing in direction of measured acceleration"""
y_pred, sigma = gp.predict(Xacc, return_std=True)
#scale up for visability
acc = Data.normalizeAcc(Yacc)*10
for i, x in enumerate(Xacc.flatten()):
    if i % 10 != 0:
        continue
    a = Arrow3D([x, x+acc[i,0]],
            [y_pred[i,0], y_pred[i,0] + acc[i,1]],
            [y_pred[i,1], y_pred[i,1] + acc[i,2]],
            mutation_scale=100, lw=1, color="b",
            arrowstyle="-|>,head_length=0.05,head_width=0.02")
    ax.add_artist(a)

"""add big arrow in the main direction of the acceleration"""
a = Arrow3D([numpy.mean(Xgps), numpy.mean(Xgps)+10*numpy.mean(Yacc[:,0])],
            [numpy.mean(Ygpsnorm[:,0]), numpy.mean(Ygpsnorm[:,0])+10*numpy.mean(Yacc[:,1])],
            [numpy.mean(Ygpsnorm[:,1]), numpy.mean(Ygpsnorm[:,1])+10*numpy.mean(Yacc[:,2])],
            mutation_scale=20, lw=3, color="g",
            arrowstyle="->,head_length=0.2,head_width=0.2")
ax.add_artist(a)

"""add arrows pointing in direction of interpolated acceleration"""
kernel = RBF(1, (0.01, 3)) * ConstantKernel(1.0, (0.1, 10)) + WhiteKernel(noise_level=1.5 ** 2)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
gp.fit(Xacc, Data.normalizeAcc(Yacc))
y_pred, sigma = gp.predict(Xgps, return_std=True)
#scale up for visability
y_pred = y_pred*10
for i, x in enumerate(Xgps.flatten()):
    a = Arrow3D([x, x+y_pred[i,0]],
            [Ygpsnorm[i,0], Ygpsnorm[i,0] + y_pred[i,1]],
            [Ygpsnorm[i,1], Ygpsnorm[i,1] + y_pred[i,2]],
            mutation_scale=100, lw=1, color="r",
            arrowstyle="-|>,head_length=0.05,head_width=0.02")
    ax.add_artist(a)

ax.set_xlabel('$a_{up}$ in $10^{-1}ms^{-2}$ and $t$ in s')
ax.set_ylabel('$a_{forward}$ in $10^{-1}ms^{-2}$ and $x$ in m')
ax.set_zlabel('$a_{right}$ in $10^{-1}ms^{-2}$ and $y$ in m')

#pyplot.zlabel('$y$ in $m$')
ax.legend()

plt.show()