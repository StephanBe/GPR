# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:29:29 2017

@author: Stephan
"""
import gmplot
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
import Data
from matplotlib import pyplot as plt

def plot():
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
    x = np.atleast_2d(np.linspace(min(Xgps), max(Xgps), 1000)).T
    #RBF is a radial kernel with a length scale parameter controlling the distance
    #   of two points influencing each other (correlation).
    #ConstantKernel defines a constant.
    #DotProduct is a linear kernel with a noise level parameter.
    kernel = RBF(0.01, (1e-3, 3)) *\
             ConstantKernel(0.001, (1e-3, 10)) +\
             WhiteKernel(1e-7, (1e-13, 1e-9))# +\
             #ConstantKernel(1, (1, 10000)) *\
             #DotProduct(0, (0, 180))
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
    ax.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred[:,0] - 1.9600 * sigma,
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
    #gmap.polygon(np.concatenate((Ygps[:,0]-0.001, Ygps[:,0]+0.001)), np.concatenate((Ygps[:,1]+0.001,Ygps[:,1]-0.001)))
    
    """draw it and change to sattelite view"""
    fileName = "googlemapsplot"+str(i)+".html"
    print("writing result to "+fileName)
    gmap.draw(fileName)
    with open(fileName, 'r') as f:
        s = f.read()
    with open(fileName, 'w') as f:
        s = s.replace("MapTypeId.ROADMAP", "MapTypeId.SATELLITE")
        f.write(s)
        
def plotFusion(fileName = "googlemapsplot-1.html"):
    from vectorRotation import rotatedAcceleration
    from SensorFusionGP import initialValues
    from TestingIntegrationMatrices import derivative_GP
    from sklearn.utils import check_random_state
    from scipy.integrate import cumtrapz
    
# =============================================================================
#     Xgps = Data.Xgps
#     Ygps = Data.Ygps
#     p = Data.latlonToMeter(Data.Ygps)
#     t_p = Data.Xgps
#     v0, forward, moving = initialValues(Data.Xacc, Data.Xgps, p[:,0], p[:,1])
#     p0 = np.array([0.,0.])
#     a = rotatedAcceleration(Data.Xacc, Data.Yacc, v0[0], v0[1], forward)
#     t_a = Data.Xacc
# =============================================================================
    
    t_a  = Data.Xacc
    t_p  = Data.Xgps
    Ygps = Data.Ygps
    t_p  = np.concatenate((Data.Xgps[:10,:], Data.Xgps[24:,:]))
    Xgps = t_p
    Ygps = np.concatenate((Data.Ygps[:10,:], Data.Ygps[24:,:]))
    #for 1st data set:
    v0 = np.array([22.14242341, -2.17809353])
    p0 = np.array([18.97244865, -5.11213311])
    forward = np.array([23.83197455, -6.74910446])
    Yacc = Data.Yacc + [0, -0.42233391105525775, 0.5373903677236002]
# =============================================================================
#     #for 2nd data set:
#     #Ygps = np.r_[Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[:1,:], Data.Ygps[1:,:]]
#     #t_p  = np.r_[Data.Xgps[:1,:], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], Data.Xgps[1:,:]]
#     Xgps = t_p
#     v0 = np.array([0.48145699, 1.50722511])
#     p0 = np.array([2.42586054, 3.584679])
#     forward = np.array([0.78791583, 2.29049936])
#     Yacc = Data.Yacc + [0, -0.22275387529091126, 0.0113005181983536]
# =============================================================================
    
    a = rotatedAcceleration(t_a, Yacc, v0[0], v0[1], forward)
    p = Data.latlonToMeter(Ygps)
    
    """plot Google Maps with GPS"""
    gmap = gmplot.GoogleMapPlotter(
            sum(Ygps[:,0])/float(len(Ygps[:,0])), 
            sum(Ygps[:,1])/float(len(Ygps[:,1])), 18,
            'AIzaSyC2I6z5RX44ZDn5z1-PiVFoEIIEVp5scKI')
    
    """ plot GPS-GPR-Prediction"""
    #add smoothed gps data
    s, l, s2, noiseGPS, noiseAcc = 20., 2., 1., 4., 4.
    
    t, lat, err_lat = derivative_GP(t_p, t_a, p[:,0].reshape(-1,1), a[:,1].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
    t, lon, err_lon = derivative_GP(t_p, t_a, p[:,1].reshape(-1,1), a[:,0].reshape(-1,1), s, l, s2, noiseGPS, noiseAcc)
    #plot some samples from posterior
    rng = check_random_state(0)
    samples_lat = rng.multivariate_normal(lat.flatten(), err_lat, 100).T
    samples_lon = rng.multivariate_normal(lon.flatten(), err_lon, 100).T
    for s in range(100):
        sl = Data.meterToLatlon(np.c_[samples_lat[:,s], samples_lon[:,s]],
                                Ygps[0,0], Ygps[0,1])
        gmap.plot(sl[:,0], sl[:,1], 'orange', edge_width=1, edge_alpha=0.5)
    
    #plot gps data
    latlon = Data.meterToLatlon(np.c_[lat, lon], Ygps[0,0], Ygps[0,1])
    #gmap.plot(Ygps[:,0], Ygps[:,1], 'cornflowerblue', edge_width=3)
    gmap.scatter(Data.Ygps[:,0], Data.Ygps[:,1], face_alpha=1, face_color='white', edge_color='lightgreen', marker=False, size=1)
    gmap.scatter(Ygps[:,0], Ygps[:,1], color='blue', marker=False, size=1)
    #plot mean posterior
    gmap.plot(latlon[:,0], latlon[:,1], 'red', edge_width=3)#, label=u'Prediction')
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(t, lat, 'r-', label='mean posterior')
    #ax.plot(t, samples_lat, label='posterior samples')
    ax.plot(Xgps, p[:,0], 'bo', label='gps data')
    ax.plot(t_a, cumtrapz(cumtrapz(a[:,1].reshape(-1,1), t_a, axis=0, initial=0)+v0[1],
                            t_a, axis=0, initial=0)+p0[1],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([lat.flatten() - 1.9600 * np.diag(err_lat),
                                (lat.flatten() + 1.9600 * np.diag(err_lat))[::-1]]),
                 alpha=.3, fc='b', ec='None', label='95% confidence interval')
    fig.legend()
    ax = fig.add_subplot(122)
    ax.plot(t, lon, 'r-', label='mean posterior')
    #ax.plot(t, samples_lat, label='posterior samples')
    ax.plot(Xgps, p[:,1], 'bo', label='gps data')
    ax.plot(t_a, cumtrapz(cumtrapz(a[:,0].reshape(-1,1), t_a, axis=0, initial=0)+v0[0],
                            t_a, axis=0, initial=0)+p0[0],
                   "--", color='orange', label="Int. acc. (lsq-fitted)")
    ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([lon.flatten() - 1.9600 * np.diag(err_lon),
                                (lon.flatten() + 1.9600 * np.diag(err_lon))[::-1]]),
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
    #gmap.polygon(np.concatenate((Ygps[:,0]-0.001, Ygps[:,0]+0.001)), np.concatenate((Ygps[:,1]+0.001,Ygps[:,1]-0.001)))
    
    """draw it and change to sattelite view"""
    print("writing result to "+fileName)
    gmap.draw(fileName)
    with open(fileName, 'r') as f:
        s = f.read()
    with open(fileName, 'w') as f:
        s = s.replace("MapTypeId.ROADMAP", "MapTypeId.SATELLITE")
        f.write(s)

if __name__ == "__main__":
    #plot()
    plotFusion()