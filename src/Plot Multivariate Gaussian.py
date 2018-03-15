# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:39:18 2018

@author: Stephan
"""

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider
from scipy.spatial.distance import pdist, squareform


def kernel(a, b, s=1, l=1):
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2,1) - 2*(a @ b.T)
    return s**2 * np.exp(-1/(2 * l**2) * sqdist)

def plot_gp_prio_samples(s=5, l=0.1):
    Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
    K_ = kernel(Xtest, Xtest, s, l)
    L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
    #f_prior = np.dot(L, np.random.normal(size=(n,10)))
    f_prior = L @ np.random.normal(size=(n,10))    
    pyplot.plot(Xtest,f_prior)
    pyplot.ylim(-10, 10)
    pyplot.title('Squared Exponential Kernel\n(sigma={}, length_scale={})'.format(s, l))
    
def mu(x):
    X = np.atleast_2d(x)
    return np.zeros(X.shape)

n = 1000

pyplot.figure()
pyplot.subplot(221)
plot_gp_prio_samples(1, 0.5)
pyplot.subplot(222)
plot_gp_prio_samples(1, 2)
pyplot.subplot(223)
plot_gp_prio_samples(3, 0.5)
pyplot.subplot(224)
plot_gp_prio_samples(3, 2)
pyplot.show()



noise=0.2
def plot_gp(ndata=0, ax=None):
    if ax == None:
        ax = pyplot.figure().add_subplot(111)
    x=np.array([[-4.0],[0.0],[1.0],[2.0],[3.0]])[:ndata,:]
    f=np.array([[-2],[1],[2],[2],[0.5]])[:ndata,:]
    s = 1
    l = 2
    K = kernel(x, x, s, l)+np.eye(ndata)*noise
    x_star = np.linspace(-5, 5, n).reshape(-1, 1)
    K_star = kernel(x, x_star, s, l)
    K_star_star = kernel(x_star, x_star, s, l)+np.eye(n)*noise
    mu_star = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f - mu(x))
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
    ax.plot(x_star, mu_star, label='prediction')
    ax.fill(np.concatenate([x_star, x_star[::-1]]),
             np.concatenate([mu_star - 1.9600 * sigma_star,
                            (mu_star + 1.9600 * sigma_star)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% confidence interval')
    ax.plot(x, f, ".", label='data')
    ax.set_ylim(-3,3)
    ax.set_title('Posterior transformed by data (slider)')
fig = pyplot.figure()
ax = fig.add_subplot(111)
axdata = fig.add_axes([0.38,0.02,0.50,0.03])
sdata = Slider(axdata, 'number of data points', valmin=0, valmax=5, valinit=0, valfmt='%0.0f')
plot_gp(0, ax)
def update(val):
    ndata = int(round(sdata.val))
    ax.clear() #workaround becaue I could not update the "fill" part
    plot_gp(ndata, ax)
    fig.canvas.draw_idle()
sdata.on_changed(update)

"""
aufloesungGauss = 5
l = n**(0.5)*4
s = 10
noise = 0
x = np.atleast_2d(np.linspace(0, 100, n)).T #x vector
y = np.atleast_2d(np.linspace(-10, 10, aufloesungGauss)).T #y vector

#repeat x and y
#x,y = np.meshgrid(x, y) #repeat x and f n times in row/column direction
#m = np.atleast_2d(np.mean(y, axis=0)) #mean vector
m = 0

#get a Kernel
pairwise_dists = squareform(pdist(np.atleast_2d(np.linspace(0, 100, n)).T, 'sqeuclidean'))
kronecker_delta = np.zeros((n,n))
i = np.arange(n)
kronecker_delta[i,i] = 1
K = s**2 * np.exp(-pairwise_dists / l ** 2) + noise * kronecker_delta


plot = pyplot.figure()

#Gaussian
#f = np.concatenate((x, y), axis=1)
f = np.atleast_2d(np.random.normal(0, s, n)).T
G = (2*np.pi)**(-n/2) * np.linalg.det(K)**(-1/2) * np.exp(-1/2*(f-m).T @ np.linalg.inv(K) @ (f-m))
p = plot.add_subplot(111)
p.plot(x, G, "gray")



#plot acceleration
p = pyplot.figure()
ax = p.add_subplot(111, projection='3d')
#ax.title("Multivariate Gaussian")
ax.plot_surface(X=x, Y=y, Z=G)
"""