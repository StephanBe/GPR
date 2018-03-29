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

def plot_gp(x=np.array([[-4.0],[0.0],[1.0],[2.0],[3.0]]),
            f=np.array([[-2],[1],[2],[2],[0.5]]), ndata=1,
            s=3, l=10, noise=1, ax=None, normalize=True):
    if ax == None:
        ax = pyplot.figure().add_subplot(111)
    meanf = None
    stdf = None
    minx = min(x)
    maxx = max(x)
    minf = min(f)
    maxf = max(f)
    if normalize:
        meanf = np.mean(f, axis=0)
        stdf = np.std(f, axis=0)
        f = (f-meanf)/stdf
    x=x[:ndata,:]
    f=f[:ndata,:]
    K = kernel(x, x, s, l)+np.eye(ndata)*noise
    #x_star = np.linspace(minx-0.1*abs(minx), maxx+0.1*abs(maxx), n).reshape(-1, 1)
    x_star = np.linspace(minx, maxx, n).reshape(-1, 1)
    K_star = kernel(x, x_star, s, l)
    K_star_star = kernel(x_star, x_star, s, l)+np.eye(n)*noise
    mu_star = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f - mu(x))
    #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
    if normalize:
        mu_star = mu_star*stdf + meanf
        f = f*stdf + meanf
    sigma_star = np.diagonal(sigma_star).reshape(-1,1)#test
    ax.plot(x_star, mu_star, label='prediction')
    ax.fill(np.concatenate([x_star, x_star[::-1]]),
             np.concatenate([mu_star - 1.9600 * sigma_star,
                            (mu_star + 1.9600 * sigma_star)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% confidence interval')
    ax.plot(x, f, ".", label='data')
    ax.set_ylim(minf-0.3*abs(maxf-minf),maxf+0.3*abs(maxf-minf))
    ax.set_title('Posterior transformed by data (slider)')

fig = pyplot.figure()
fig.subplots_adjust(bottom=0.2)
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
axdata = fig.add_axes([0.38,0.14,0.50,0.03])
axsigma = fig.add_axes([0.38,0.09,0.50,0.03])
axlength = fig.add_axes([0.38,0.05,0.50,0.03])
axnoise = fig.add_axes([0.38,0.01,0.50,0.03])
gps = Data.Ygps#Data.latlonToMeter(Data.Ygps)
gpsy = gps[:,0].reshape(-1,1) #lat
gpsx = gps[:,1].reshape(-1,1) #lon
#==============================================================================
# s = 90000
# l = 60
# noise=9
#==============================================================================
s=0.1
l=10
noise=0.0001
sdata = Slider(axdata, 'number of data points', valmin=1, valmax=len(gpsx), valinit=len(gpsx), valfmt='%0.0f')
ssigma = Slider(axsigma, 'sigma', valmin=0.01, valmax=10, valinit=s, valfmt='%0.3f')
slength = Slider(axlength, 'length scale', valmin=1, valmax=100.0, valinit=l, valfmt='%0.4f')
snoise = Slider(axnoise, 'noise', valmin=0.000001, valmax=0.0001, valinit=noise, valfmt='%0.5f')
plot_gp(Data.Xgps, gpsy, len(gpsx), s, l, noise, ax)
#plot_gp2(Data.Xgps, gps, len(gpsx), 90000, 60, 9, ax)
def update(val):
    ndata = int(round(sdata.val))
    s = ssigma.val
    l = slength.val
    noise = snoise.val
    ax.clear() #workaround becaue I could not update the "fill" part
    plot_gp(Data.Xgps, gpsy, ndata, s, l, noise, ax)
    #plot_gp2(Data.Xgps, gps, ndata, s, l, noise, ax)
    fig.canvas.draw_idle()
sdata.on_changed(update)
ssigma.on_changed(update)
slength.on_changed(update)
snoise.on_changed(update)


def plot_gp2(x=np.array([[0.0],[1.04],[1.08],[1.12],[3.0]]),
            f=np.array([[0,0],[0.1,-0.1],[1,0],[3,1],[5,1]]), ndata=1,
            s=3.0, l=10.0, noise=0, ax=None, normalize=True):
    if ax == None:
        fig= pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(bottom=0.2)
    meanf = None
    stdf = None
    minx = min(x)
    maxx = max(x)
    minf = np.min(f, axis=0)
    maxf = np.max(f, axis=0)
    if normalize:
        meanf = np.mean(f, axis=0)
        stdf = np.std(f, axis=0)
        f = (f-meanf)/stdf
    x=x[:ndata,:]
    f=f[:ndata,:]
    f1 = np.atleast_2d(f[:,0]).T
    f2 = np.atleast_2d(f[:,1]).T
    K = kernel(x, x, s, l)+np.eye(ndata)*noise
    #x_star = np.linspace(minx-0.1*abs(minx), maxx+0.1*abs(maxx), n).reshape(-1, 1)
    x_star = np.linspace(minx, maxx, n).reshape(-1, 1)
    K_star = kernel(x, x_star, s, l)
    K_star_star = kernel(x_star, x_star, s, l)+np.eye(n)*noise
    mu_star1 = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f1 - mu(x))
    mu_star2 = mu(x_star) + K_star.T @ np.linalg.inv(K) @ (f2 - mu(x))
    #mu_star = mymu(x_star, x, f) + K_star.T @ np.linalg.inv(K) @ (f - mymu(x, x, f))
    if normalize:
        mu_star1 = mu_star1*stdf + meanf
        mu_star2 = mu_star2*stdf + meanf
        f = f*stdf + meanf
    sigma_star = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
    sigma_star = np.diagonal(sigma_star).reshape(-1,1)#test
    
    """plot"""
    ax.plot(x_star.flatten(), mu_star1.flatten(), mu_star2.flatten(), label='prediction')

#==============================================================================
#     ax.fill(np.concatenate([mu_star1, mu_star1[::-1]]),
#              np.concatenate([mu_star2 - 1.9600 * sigma_star,
#                             (mu_star2 + 1.9600 * sigma_star)[::-1]]),
#              alpha=.3, fc='b', ec='None', label='95% confidence interval')
#==============================================================================
    ax.plot(x, f[:,0], f[:,1], ".", label='data')
    ax.set_xlim(minx[0]-0.3*abs(maxx[0]-minx[0]),maxx[0]+0.3*abs(maxx[0]-minx[0]))
    ax.set_ylim(minf[0]-0.3*abs(maxf[0]-minf[0]),maxf[0]+0.3*abs(maxf[0]-minf[0]))
    ax.set_zlim(minf[1]-0.3*abs(maxf[1]-minf[1]),maxf[1]+0.3*abs(maxf[1]-minf[1]))
    ax.invert_yaxis()
    ax.set_xlabel(u'time in $s$')
    ax.set_ylabel(u'latitude in $°$')
    ax.set_zlabel(u'longitude in $°$')
    ax.legend()
    ax.set_title('Posterior transformed by data (slider)')

fig2 = pyplot.figure()
fig2.subplots_adjust(bottom=0.2)
ax2 = fig2.add_subplot(111, projection='3d')
#ax = fig2.add_subplot(111)
axdata2 = fig2.add_axes([0.38,0.14,0.50,0.03])
axsigma2 = fig2.add_axes([0.38,0.09,0.50,0.03])
axlength2 = fig2.add_axes([0.38,0.05,0.50,0.03])
axnoise2 = fig2.add_axes([0.38,0.01,0.50,0.03])
gps = Data.Ygps#Data.latlonToMeter(Data.Ygps)
gpsy = gps[:,0].reshape(-1,1) #lat
gpsx = gps[:,1].reshape(-1,1) #lon
#==============================================================================
# s = 90000
# l = 60
# noise=9
#==============================================================================
s=0.1
l=10
noise=0.0001
sdata2 = Slider(axdata2, 'number of data points', valmin=1, valmax=len(gpsx), valinit=len(gpsx), valfmt='%0.0f')
ssigma2 = Slider(axsigma2, 'sigma', valmin=0.01, valmax=10, valinit=s, valfmt='%0.3f')
slength2 = Slider(axlength2, 'length scale', valmin=1, valmax=100.0, valinit=l, valfmt='%0.4f')
snoise2 = Slider(axnoise2, 'noise', valmin=0.000001, valmax=0.0001, valinit=noise, valfmt='%0.5f')
#plot_gp(Data.Xgps, gpsy, len(gpsx), s, l, noise, ax)
plot_gp2(Data.Xgps, gps, len(gpsx), s, l, noise, ax2)
def update2(val):
    ndata = int(round(sdata2.val))
    s = ssigma2.val
    l = slength2.val
    noise = snoise2.val
    ax.clear() #workaround becaue I could not update the "fill" part
    #plot_gp(Data.Xgps, gpsy, ndata, s, l, noise, ax)
    plot_gp2(Data.Xgps, gps, ndata, s, l, noise, ax2)
    fig2.canvas.draw_idle()
sdata2.on_changed(update2)
ssigma2.on_changed(update2)
slength2.on_changed(update2)
snoise2.on_changed(update2)

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