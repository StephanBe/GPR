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

i=-1#for i in range(2, 9):

Xgps = Data.Xgps
Ygps = Data.Ygps

"""plot Google Maps with GPS"""
gmap = gmplot.GoogleMapPlotter(
        sum(Ygps[:,0])/float(len(Ygps[:,0])), 
        sum(Ygps[:,1])/float(len(Ygps[:,1])), 18,
        'AIzaSyC2I6z5RX44ZDn5z1-PiVFoEIIEVp5scKI')
gmap.plot(Ygps[:,0], Ygps[:,1], 'cornflowerblue', edge_width=3)
gmap.scatter(Ygps[:,0], Ygps[:,1], 'blue', marker=False, size=1)

""" plot GPS-GPR-Prediction"""
#add smoothed gps data
x = numpy.atleast_2d(numpy.linspace(min(Xgps), max(Xgps), 1000)).T
#RBF is a radial kernel with a length scale parameter controlling.
#DotProduct is a linear kernel with a noise level parameter.
kernel = RBF(0.1, (0.01, 1)) *\
         ConstantKernel(1.0, (0.1, 10)) +\
         DotProduct(0.001, (0.001, 0.01)) *\
         ConstantKernel(1.0, (0.01, 2.0))
#noise = 0.0001
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
mean = Ygps.mean(0)
gp.fit(Xgps, Ygps-mean)
y_pred, sigma = gp.predict(x, return_std=True)
gmap.plot(y_pred[:,0]+mean[0], y_pred[:,1]+mean[1], c='red', edge_width=1)#, label=u'Prediction')
for s in range(10):
    sample = gp.sample_y(x, n_samples=len(x), random_state=s)
    gmap.plot(sample[:,0]+mean[0], sample[:,1]+mean[1], c='orange', edge_width=0.1)
gmap.polygon(numpy.concatenate((Ygps[:,0]-0.001, Ygps[:,0]+0.001)), numpy.concatenate((Ygps[:,1]+0.001,Ygps[:,1]-0.001)))

"""draw it and change to sattelite view"""
fileName = "googlemapsplot"+str(i)+".html"
print("writing result to "+fileName)
gmap.draw(fileName)
with open(fileName, 'r') as f:
    s = f.read()
with open(fileName, 'w') as f:
    s = s.replace("MapTypeId.ROADMAP", "MapTypeId.SATELLITE")
    f.write(s)
                