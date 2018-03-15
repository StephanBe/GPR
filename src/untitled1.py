# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 06:20:42 2017

@author: Stephan
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot

x = np.arange(1,51)
y = np.arange(1,51)
X, Y = np.meshgrid(x, y)

points = zip(obs_x,  obs_y)
values = obs_data    # Replace with your observed data

gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.001)
gp.fit(points, values)
XY_pairs = np.column_stack([X.flatten(), Y.flatten()])
predicted = gp.predict(XY_pairs).reshape(X.shape)
pyplot.plot(predicted)