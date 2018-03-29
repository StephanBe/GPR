# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:29:29 2017

@author: Stephan
"""
import numpy
from matplotlib import pyplot

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import Data

def gpr(x_train, y_train, x_pred):
    #WhiteKernel for noise estimation (alternatively set alpha in GaussianProcessRegressor()) 
    #ConstantKernel for signal variance
    #RBF for length-scale
    kernel = RBF(0.1, (0.01, 10))*ConstantKernel(1.0, (0.1, 100)) + WhiteKernel(0.1, (0.01,1))
    #noise = 0.1
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    mean = numpy.mean(y_train)
    gp.fit(x_train, y_train-mean)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    y_pred += mean
    return y_pred, sigma

Xacc = Data.Xacc
Yacc = Data.Yacc

"""plot cumulative distribution"""
cumDistX = numpy.sort(Yacc[:,0])-numpy.median(Yacc[:,0])
cumDistY = numpy.sort(Yacc[:,1])-numpy.median(Yacc[:,1])
cumDistZ = numpy.sort(Yacc[:,2])-numpy.median(Yacc[:,2])
cumX = numpy.arange(len(Yacc))/len(Yacc)
cumY = numpy.copy(cumX)
cumZ = numpy.copy(cumX)

pyplot.subplot(221)
pyplot.title("Cumulative Distribution Function\nOf Acceleration Data")
pyplot.plot(cumDistX, cumX, 'r-', label="$a_x$")
pyplot.plot(cumDistY, cumY, 'g-', label="$a_y$")
pyplot.plot(cumDistZ, cumZ, 'b-', label="$a_z$")
pyplot.xlabel("$a$ in $m/s^2$")
pyplot.ylabel("")
pyplot.legend()

"""plot distribution"""
selectionX = numpy.append(numpy.diff(cumDistX) != 0, False)
selectionY = numpy.append(numpy.diff(cumDistY) != 0, False)
selectionZ = numpy.append(numpy.diff(cumDistZ) != 0, False)
distX = numpy.diff(cumX[selectionX])/numpy.diff(cumDistX[selectionX])
distY = numpy.diff(cumY[selectionY])/numpy.diff(cumDistY[selectionY])
distZ = numpy.diff(cumZ[selectionZ])/numpy.diff(cumDistZ[selectionZ])
distX /= sum(distX)
distY /= sum(distY)
distZ /= sum(distZ)

pyplot.subplot(223)
pyplot.title("Distribution Function\nOf Acceleration Data")
pyplot.plot(cumDistX[selectionX][:-1], distX, 'r-', label="$a_x$")
pyplot.plot(cumDistY[selectionY][:-1], distY, 'g-', label="$a_y$")
pyplot.plot(cumDistZ[selectionZ][:-1], distZ, 'b-', label="$a_z$")
pyplot.xlabel("$a$ in $m/s^2$")
pyplot.ylabel("")
pyplot.legend()

"""plot acceleration"""
pyplot.subplot(122)
pyplot.title("Acceleration Data With GPR-Prediction")

"""GP regression acceleration x"""
x = numpy.atleast_2d(numpy.linspace(-5, 35, 10000)).T

y_pred, sigma = gpr(Xacc, Yacc[:,0], x)
pyplot.plot(x, y_pred, 'r--', label=u'Prediction $a_x$', linewidth=0.5)
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.3, fc='r', ec='None', label='95% confidence interval')
#original data
pyplot.plot(Xacc, Yacc[:,0], 'r-', label=u'$a_x$', linewidth=0.5)

"""GP regression acceleration y"""
y_pred, sigma = gpr(Xacc, Yacc[:,1], x)
pyplot.plot(x, y_pred, 'g--', label=u'Prediction $a_y$', linewidth=0.5)
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.3, fc='g', ec='None', label='95% confidence interval')
#original data
pyplot.plot(Xacc, Yacc[:,1], 'g-', label=u'$a_y$', linewidth=0.5)

"""GP regression acceleration z"""
y_pred, sigma = gpr(Xacc, Yacc[:,2], x)
pyplot.plot(x, y_pred, 'b--', label=u'Prediction $a_z$', linewidth=0.5)
pyplot.fill(numpy.concatenate([x, x[::-1]]),
         numpy.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.3, fc='b', ec='None', label='95% confidence interval')
#original data
pyplot.plot(Xacc, Yacc[:,2], 'b-', label=u'$a_z$', linewidth=0.5)

pyplot.xlabel('$t$ in $s$')
pyplot.ylabel('$a$ in $m/s^2$')
pyplot.ylim(min(Yacc.ravel())-1, max(Yacc.ravel())+1)
#pyplot.xlim(19,21)
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.show()