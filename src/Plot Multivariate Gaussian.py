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
import Data

#Squared Exponential Kernel

def sqared_exponential_kernel(a, b, s=1, l=1):
    """
    Squared Exponential Kernel
    """
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2,1) - 2*(a @ b.T)
    return s**2 * np.exp(-1/(2 * l**2) * sqdist)

#Linear Kernel

def linear_kernel(x1, x2, s1=0, s2=1, c=0):
    """
    Squared Exponential Kernel
    c is the starting point for the linear prior.
    """
    return s1**2 + (s2**2) * (x1-c)*(x2-c).T

def kernel(a, b, s=1, l=1):
    return sqared_exponential_kernel(a, b, s, l) + linear_kernel(a, b)

def plot_gp_prio_samples(s=5, l=0.1):
    Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
    K_ = sqared_exponential_kernel(Xtest, Xtest, s, l)
    L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
    #f_prior = np.dot(L, np.random.normal(size=(n,10)))
    f_prior = L @ np.random.normal(size=(n,10))    
    pyplot.plot(Xtest,f_prior)
    pyplot.ylim(-10, 10)
    pyplot.title('Squared Exponential Kernel\n(sigma={}, length_scale={})'.format(s, l))
    
def mu(x):
    X = np.atleast_2d(x)
    return np.zeros(X.shape)

def mymu(x_star, x, f):
    if np.array_equal(x_star, x):
        return np.atleast_2d(f)
    else:
        #in Treppenmanier die Werte von f fortsetzen für x_star, solange
        #bis f einen neuen wert hat
        m = np.zeros(np.atleast_2d(x_star).shape)
        stepBegin = 0
        step = -1
        for step in range(0, len(x.flatten())-1):
            stepEnd = stepBegin+1
            m[stepBegin,0] = f[step,0]
            while x_star[stepEnd,0] < x[step+1,0]:
                m[stepEnd,0] = f[step,0]
                stepEnd += 1
            stepBegin = stepEnd
        for i in range(stepBegin, len(x_star.flatten())):
            m[i] = f[step+1, 0]
        return np.atleast_2d(m)


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



noise=1
def plot_gp(x=np.array([[-4.0],[0.0],[1.0],[2.0],[3.0]]),
            f=np.array([[-2],[1],[2],[2],[0.5]]), ndata=1, ax=None):
    if ax == None:
        ax = pyplot.figure().add_subplot(111)
    s = 3
    l = 10
    minx = min(x)
    maxx = max(x)
    minf = min(f)
    maxf = max(f)
    x=x[:ndata,:]
    f=f[:ndata,:]
    K = kernel(x, x, s, l)+np.eye(ndata)*noise
    x_star = np.linspace(minx-0.1*abs(minx), maxx+0.1*abs(maxx), n).reshape(-1, 1)
    K_star = kernel(x, x_star, s, l)
    K_star_star = kernel(x_star, x_star, s, l)+np.eye(n)*noise
    mu_star = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f - mu(x))
    #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
    ax.plot(x_star, mu_star, label='prediction')
    sigma_star = np.diagonal(sigma_star).reshape(-1,1)#test
    ax.fill(np.concatenate([x_star, x_star[::-1]]),
             np.concatenate([mu_star - 1.9600 * sigma_star,
                            (mu_star + 1.9600 * sigma_star)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% confidence interval')
    ax.plot(x, f, ".", label='data')
    ax.set_ylim(minf-0.1*abs(minf),maxf+0.1*abs(maxf))
    ax.set_title('Posterior transformed by data (slider)')
fig = pyplot.figure()
ax = fig.add_subplot(111)
axdata = fig.add_axes([0.38,0.02,0.50,0.03])
gps = Data.latlonToMeter(Data.Ygps)
gpsx = gps[:,0].reshape(-1,1)
gpsy = gps[:,1].reshape(-1,1)
sdata = Slider(axdata, 'number of data points', valmin=1, valmax=len(gpsx), valinit=5, valfmt='%0.0f')
plot_gp(gpsx, gpsy, 5, ax)
def update(val):
    ndata = int(round(sdata.val))
    ax.clear() #workaround becaue I could not update the "fill" part
    plot_gp(gpsx, gpsy, ndata, ax)
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