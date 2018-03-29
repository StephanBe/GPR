# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:29:29 2017

@author: Stephan
"""
import gmplot
import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
import Data
from matplotlib import pyplot as plt

i=-1#for i in range(2, 9):

Xgps = Data.Xgps
Ygps = Data.Ygps

"""plot Google Maps with GPS"""
gmap = gmplot.GoogleMapPlotter(
        sum(Ygps[:,0])/float(len(Ygps[:,0])), 
        sum(Ygps[:,1])/float(len(Ygps[:,1])), 18,
        'AIzaSyC2I6z5RX44ZDn5z1-PiVFoEIIEVp5scKI')

""" plot GPS-GPR-Prediction"""
#add smoothed gps data
x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
#RBF is a radial kernel with a length scale parameter controlling the distance
#   of two points influencing each other (correlation).
#ConstantKernel defines a constant.
#DotProduct is a linear kernel with a noise level parameter.
kernel = RBF(10, (1, 50)) *\
         ConstantKernel(0.001, (0.0001, 0.1)) +\
         WhiteKernel(0.001, (0.0001, 0.1)) +\
         ConstantKernel(0.01, (0.0001, 0.01)) *\
         DotProduct(0.001, (0.001, 0.01))
#noise = 0.0001
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
gp.fit(Xgps, Ygps)
y_pred, sigma = gp.predict(x, return_std=True)
#plot some samples from posterior
sample = gp.sample_y(x, n_samples=10)
for s in range(10):
    gmap.plot(sample[:,0,s], sample[:,1,s], 'orange', edge_width=1)

#plot gps data
gmap.plot(Ygps[:,0], Ygps[:,1], 'cornflowerblue', edge_width=3)
gmap.scatter(Ygps[:,0], Ygps[:,1], 'blue', marker=False, size=1)
#plot mean posterior
gmap.plot(y_pred[:,0], y_pred[:,1], 'red', edge_width=3)#, label=u'Prediction')

fig, ax = plt.subplots()
ax.plot(x, y_pred[:,0], 'r-', label='mean posterior')
ax.plot(Xgps, Ygps[:,0], 'bo', label='original data')
ax.fill(numpy.concatenate([x, x[::-1]]),
             numpy.concatenate([y_pred[:,0] - 1.9600 * sigma,
                            (y_pred[:,0] + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% confidence interval')
fig.show()
#==============================================================================
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
# mean = Ygps.mean(0)
# std = (Ygps-mean).std(0)
# gp.fit(Xgps, (Ygps-mean)/std)
# y_pred, sigma = gp.predict(x, return_std=True)
# #plot mean posterior
# gmap.plot((y_pred[:,0]*std[0]+mean[0]), (y_pred[:,1]*std[1]+mean[1]), 'red', edge_width=1)#, label=u'Prediction')
# #plot some samples from posterior
# sample = gp.sample_y(x, n_samples=10)
# for s in range(10):
#     gmap.plot((sample[:,0,s]*std[0]+mean[0]), (sample[:,1,s]*std[1]+mean[1]), 'orange', edge_width=1)
# 
#==============================================================================
#gmap.polygon(numpy.concatenate((Ygps[:,0]-0.001, Ygps[:,0]+0.001)), numpy.concatenate((Ygps[:,1]+0.001,Ygps[:,1]-0.001)))

"""draw it and change to sattelite view"""
fileName = "googlemapsplot"+str(i)+".html"
print("writing result to "+fileName)
gmap.draw(fileName)
with open(fileName, 'r') as f:
    s = f.read()
with open(fileName, 'w') as f:
    s = s.replace("MapTypeId.ROADMAP", "MapTypeId.SATELLITE")
    f.write(s)
                