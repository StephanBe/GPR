#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:43:28 2018

@author: stephan
"""

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, cos, sin
from matplotlib.widgets import Slider
#from sklearn.preprocessing import normalize

def rotate(a, r):
    normY = r / sqrt(r.T @ r)
    normX = [normY[1], -normY[0]]
    b = np.array([normX, normY]).T @ a
    return(b)

if __name__ == "__main__":
    aAngle = 2
    rAngle = 4
    a=np.array([cos(aAngle), sin(aAngle)])
    r=np.array([2*cos(rAngle), 2*sin(rAngle)])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.2)
    axvec = fig.add_axes([0.18,0.08,0.70,0.03])
    axrot = fig.add_axes([0.18,0.02,0.70,0.03])
    svec = Slider(axvec, 'vector', valmin=-5.0, valmax=5.0, valinit=2.0, valfmt='%0.2f')
    srot = Slider(axrot, 'new basis', valmin=-5.0, valmax=5.0, valinit=4.0, valfmt='%0.2f')
    
    def vectorplot(a, style="r-", label=""):
        ax.plot([0,a[0]], [0,a[1]], style, label=label)
    
    vectorplot(a, "r-", label="vector")
    vectorplot(r, "g--", label="new basis")
    vectorplot((r / sqrt(r.T @ r) / 2), "b--", label="new basis normalized")
    vectorplot(([(r / sqrt(r.T @ r) / 2)[1], -(r / sqrt(r.T @ r) / 2)[0]]), "b--",
               label="new basis normalized")
    vectorplot(rotate(a, r), "r--", label="rotated vector")
    m = max(max(abs(a)),max(abs(r))) + 1
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    fig.legend()
    
    def update(val):
        aAngle = svec.val
        rAngle = srot.val
        a=np.array([cos(aAngle), sin(aAngle)])
        r=np.array([2*cos(rAngle), 2*sin(rAngle)])
        ax.clear() #workaround becaue I could not update the "fill" part
        vectorplot(a, "r-", label="vector")
        vectorplot(r, "g-", label="new basis")
        vectorplot((r / sqrt(r.T @ r) / 2), "b--", label="new basis normalized")
        vectorplot(([(r / sqrt(r.T @ r) / 2)[1], -(r / sqrt(r.T @ r) / 2)[0]]), "b--",
                   label="new basis normalized")
        vectorplot(rotate(a, r), "r--", label="rotated vector")
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        fig.canvas.draw_idle()
        
    svec.on_changed(update)
    srot.on_changed(update)
    
    plt.show()
