# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:55:09 2018

@author: Stephan
"""

import numpy
import csv
from datetime import datetime
import os
from matplotlib import pyplot
from math import cos, pi, atan2
from scipy import integrate
import platform


if platform.system() == 'Windows':
    vufo_data = '..\\data\\Sondersignal\\Kritisch\\City\\20130421_142927_Bremsen'
    #vufo_data = '..\\data\\Erprobung\\Fahrsicherheitstraining\\Ausweichen Touran'
else:
    #workaround for an unsolved mounting conflict on my system
    if os.path.isdir("../data"):
        vufo_data = '../data'
    else:
        vufo_data = '../data2'
    vufo_data += '/Sondersignal/Kritisch/City/20130421_142927_Bremsen'
    #vufo_data += '/Erprobung/Fahrsicherheitstraining/Ausweichen Touran'

"""liest VUFO-CSV-Daten ein"""
def read(filename, normalizeTime):
    Y = []
    X = []
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            X.append(datetime.strptime(row[0]+' '+row[1], '%d.%m.%Y %H:%M:%S,%f'))
            Y.append([float(x.replace(',', '.')) for x in row[2:]])
    #X = numpy.asarray(X)
    """Normierung auf den Start und Konvertierung in Sekunden"""
    if normalizeTime:
        X = numpy.atleast_2d([(x-min(X)).total_seconds() for x in X]).T
    Y = numpy.asarray(Y)
    return [X, Y]

"""transforms latitude and longitude to meters"""
def latlonToMeter(latlon):
    #earths curvature depends on latitude
    l = numpy.array(latlon[:,0])
    #meters per latitude
    m_per_deg_lat = 111132.92
    m_per_deg_lat -= 559.82 * numpy.cos(2.0 * l)
    m_per_deg_lat += 1.175 * numpy.cos(4.0 * l)
    m_per_deg_lat -= 0.0023 * numpy.cos(6.0 * l)
    #meters per longitude
    m_per_deg_lon = 111412.84 * numpy.cos(l)
    m_per_deg_lon -= 93.5 * numpy.cos(3.0 * l)
    m_per_deg_lon += 0.118 * numpy.cos(5.0 * l)
    #calculate meters and normalize to start at 0m
    return (latlon-latlon[0,:])*numpy.array([m_per_deg_lat, m_per_deg_lon]).T

"""transforms meters to latitude and longitude"""
def meterToLatlon(meter, lat0, lon0):
    #earths curvature depends on latitude
    l = lat0
    #meters per latitude
    m_per_deg_lat = 111132.92
    m_per_deg_lat -= 559.82 * numpy.cos(2.0 * l)
    m_per_deg_lat += 1.175 * numpy.cos(4.0 * l)
    m_per_deg_lat -= 0.0023 * numpy.cos(6.0 * l)
    #meters per longitude
    m_per_deg_lon = 111412.84 * numpy.cos(l)
    m_per_deg_lon -= 93.5 * numpy.cos(3.0 * l)
    m_per_deg_lon += 0.118 * numpy.cos(5.0 * l)
    #calculate meters and normalize to start at 0m
    return meter / [m_per_deg_lat, m_per_deg_lon] + [lat0, lon0]

"""zieht Mittelwert der Daten ab"""   
def normalizeAcc(Yacc):
    y = [[Y[0]-numpy.mean(Yacc[:,0]),
          Y[1]-numpy.mean(Yacc[:,1]),
          Y[2]-numpy.mean(Yacc[:,2])]
          for Y in Yacc]
    return  numpy.asarray(y)


#normalisieren der zeit entfernen
"""springe in das gegebene Verzeichnis"""
prevdir=os.getcwd()
os.chdir(vufo_data)
"""Beschleunigung einlesen"""
Xacc, Yacc = read('AccelData.txt', False)
"""GPS-Daten einlesen"""
Xgps, Ygps = read('GPSData.txt', False)
"""Gyro-Daten einlesen"""
Xgyr, Ygyr = read('GyroData.txt', False)
os.chdir(prevdir)

"""Normalisiere Zeit"""
minx = min([min(Xacc), min(Xgps), min(Xgyr)])
Xacc = numpy.atleast_2d([(x-minx).total_seconds() for x in Xacc]).T
Xgps = numpy.atleast_2d([(x-minx).total_seconds() for x in Xgps]).T
Xgyr = numpy.atleast_2d([(x-minx).total_seconds() for x in Xgyr]).T

"""Korrigiere Zeitstempelfehler"""
Xacc_corrected = numpy.copy(Xacc)
Xgyr_corrected = numpy.copy(Xgyr)
delta     = Xacc[1:]-Xacc[:-1]
delta_gyr = Xgyr[1:]-Xgyr[:-1]
normal     = numpy.median(delta)
normal_gyr = numpy.median(delta_gyr)
for i in range(len(Xacc)-1):
    if Xacc_corrected[i+1]-Xacc_corrected[i] < 0.5 * normal:
        """Korrigiere Anfang eines Fehlers"""
        Xacc_corrected[i+1] = Xacc_corrected[i] + normal
for i in range(len(Xgyr)-1):
    if Xgyr_corrected[i+1]-Xgyr_corrected[i] < 0.5 * normal_gyr:
        """Korrigiere Anfang eines Fehlers"""
        Xgyr_corrected[i+1] = Xgyr_corrected[i] + normal_gyr

Xacc_corrected2 = numpy.copy(Xacc)
Xgyr_corrected2 = numpy.copy(Xgyr)
i = 0
while i < len(Xacc)-1:
    if Xacc_corrected2[i+1]-Xacc_corrected2[i] < 0.5 * normal:
        """Finde Ende des Fehlers"""
        j = i+1
        while j < len(Xacc)-1 and Xacc[j+1]-Xacc[j] < 0.5 * normal:
            j += 1
        """Korrigiere den Fehler"""
        stepWidth = (Xacc[j+1] - Xacc[i])/(j+1-i)
        for k in range(i+1, j+1):
            Xacc_corrected2[k] = Xacc[i] + (k - i) * stepWidth
        i = j
    i += 1
    
i = 0
while i < len(Xgyr)-1:
    if Xgyr_corrected2[i+1]-Xgyr_corrected2[i] < 0.5 * normal_gyr:
        """Finde Ende des Fehlers"""
        j = i+1
        while j < len(Xgyr)-1 and Xgyr[j+1]-Xgyr[j] < 0.5 * normal_gyr:
            j += 1
        """Korrigiere den Fehler"""
        stepWidth = (Xgyr[j+1] - Xgyr[i])/(j+1-i)
        for k in range(i+1, j+1):
            Xgyr_corrected2[k] = Xgyr[i] + (k - i) * stepWidth
        i = j
    i += 1
    
"""Justiere Beschleunigungsdaten"""
#Getestet mit VUFO: Erprobung/Fahrsicherheitstraining/Ausweichen Touran
#
#print(numpy.mean(Yacc[1:120,:], axis=0))
#[ 9.88905903 -0.11588741 -0.19282377]
#print(numpy.median(Yacc[1:120,:], axis=0))
#[ 9.921572  -0.1340753 -0.2106898]
#print(numpy.mean(Ygyr[0:120,:], axis=0))
#[ 0.002586   -0.00454077 -0.00711658]

diffAcc = [-9.81+9.88905903, -0.11588741, -0.19282377]
Yacc[:,0] = Yacc[:,0]-diffAcc[0]
Yacc[:,1] = Yacc[:,1]-diffAcc[1]
Yacc[:,2] = Yacc[:,2]-diffAcc[2]
diffGyr = [      0.00258600, -0.00454077, -0.00711658]
Ygyr[:,0] = Ygyr[:,0]-diffGyr[0]
Ygyr[:,1] = Ygyr[:,1]-diffGyr[1]
Ygyr[:,2] = Ygyr[:,2]-diffGyr[2]

print()
print("THE ACCELERATION VALUES ARE REDUCED BY " + str(diffAcc)+
      " AND ANGULAR VELOCITY BY "+ str(diffGyr) +
      " TO "+
      "CALIBRATE THEM LANDING ON 0 MOVEMENT AFTER 120 ENTRIES FROM DATA SET")
print("VUFO/Erprobung/Fahrsicherheitstraining/Ausweichen Touran")
print("THIS SHOULD BE DONE PROPERLY INSTEAD IN A PRODUCTIVE VERSION!")
print()

tmp = numpy.copy(Yacc)
#before: android x (Yacc[:,0] = sensor accelerates upwards)
UP = 0
YAW = 0
Yacc[:,UP] = tmp[:,0] #after: Yacc[:,0] = acceleration upwards (Erdbeschleunigung wirkt, als würde das Auto konstant nach oben beschleunigen)
#before: android z (Yacc[:,2] = sensor accelerates backwards)
FORWARD = 1
ROLL = 1
Yacc[:,FORWARD] = -tmp[:,2] #after: Yacc[:,1] = acceleration forwards
#before: android y (Yacc[:,1] = sensor accelerates leftwards)
LEFT = 2
PITCH = 2
Yacc[:,LEFT] = tmp[:,1] #after: Yacc[:,2] = acceleration leftwards
#Yacc[:,LEFT] = np.zeros(Yacc[:,LEFT].shape)

"""Plots zu den gewählten Daten und dem Zeitstempelproblem"""
if __name__ == "__main__":
    pyplot.figure()
    pyplot.subplot(231)
    pyplot.plot(Xacc, label="original")
    pyplot.plot(Xacc_corrected2, "--",
                label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected, "--",
                label="corrected by keeping a normal time frame")
    pyplot.title("Time Stamp\nOf Each Data Entry")
    pyplot.xlabel("$i$th entry")
    pyplot.ylabel("$t$ in $s$")
#==============================================================================
#     pyplot.legend()
#==============================================================================
    
    pyplot.subplot(232)
    pyplot.plot(Xacc[1:] - Xacc[:-1], label="original")
    pyplot.plot(Xacc_corrected2[1:] - Xacc_corrected2[:-1], "--",
                label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected[1:] - Xacc_corrected[:-1], "--",
                label="corrected by keeping a normal time frame")
    pyplot.title("Time Elapsed\nBetween Two Following Entries")
    pyplot.xlabel("$i$th entry")
    pyplot.ylabel("$t_{i+1}-t_{i}$ in $s$")
#==============================================================================
#     pyplot.legend()
#==============================================================================
    
    pyplot.subplot(233)
    pyplot.title("Time Difference\nOf Actual Values To Constant Intervals")
    medianDelta = [Xacc[0]+i*numpy.median(delta) for i in range(0,len(Xacc))]
    meanDelta = [Xacc[0]+i*numpy.mean(delta) for i in range(0,len(Xacc))]
    pyplot.plot(Xacc-medianDelta, label="original")
    pyplot.plot(Xacc_corrected2-medianDelta, "--",
                label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected-medianDelta, "--",
                label="corrected by keeping a normal time frame")
    pyplot.xlabel("$i$th entry")
    pyplot.ylabel("$t_{measured}-t_{artificial}$ in s")
    #pyplot.plot(medianDelta, label="median delta")
#==============================================================================
#     pyplot.legend()
#==============================================================================
    #pyplot.plot(delta, label="original")
    #pyplot.plot(Xacc_corrected[1:]-Xacc_corrected[:-1], label="corrected")
    #pyplot.legend()
    pyplot.subplot(212)
    pyplot.title("Car Acceleration to the left")
    pyplot.plot(Xacc, Yacc[:,LEFT], label="original")
    pyplot.plot(Xacc_corrected2, Yacc[:,LEFT], "--",
                label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected, Yacc[:,LEFT], "--",
                label="corrected by keeping a normal time frame")
    pyplot.xlabel("$t$ in $s$")
    pyplot.ylabel("$a$ in $m*s^{-1}$")
    pyplot.legend()
    pyplot.show()
    
    pyplot.figure()
    pyplot.subplot(111)
    pyplot.title("Car Position")
    g = latlonToMeter(Ygps)
    cb = pyplot.scatter(g[:,1], g[:,0], c=Xgps.flatten(), label="position")
    pyplot.colorbar(cb, label="$t$ in $s$")
    pyplot.xlabel("longitude in $m$")
    pyplot.ylabel("latitude in $m$")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    
    pyplot.figure()
    pyplot.subplot(311)
    pyplot.title("Raw acceleration data")
    #pyplot.grid()
    pyplot.plot(Xacc, Yacc[:,UP], linewidth=0.5, label="up")#correct
    pyplot.plot(Xacc, Yacc[:,LEFT], linewidth=0.5, label="left")#correct
    pyplot.plot(Xacc, -Yacc[:,FORWARD], linewidth=0.5, label="back")#correct
    pyplot.ylabel("$a$ in $m/s^2$")
    leg = pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    pyplot.xlim((-1,32))
    pyplot.subplot(312)
    pyplot.title("Raw angular velocity data")
    #pyplot.grid()
    pyplot.plot(Xgyr, Ygyr[:,0], linewidth=0.5, label="yaw")# (counter-clockweise around down)")#correct
    pyplot.plot(Xgyr, Ygyr[:,1], linewidth=0.5, label="pitch")# (clockwise around left)")#correct
    pyplot.plot(Xgyr, Ygyr[:,2], linewidth=0.5, label="roll")# (clockwise around backward)")#correct
    pyplot.ylabel("$\omega$ in $rad/s$")
    leg = pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    pyplot.xlim((-1,32))
    ax = pyplot.subplot(313)
    ax.set_title("Raw position data")
    #pyplot.grid()
    l1 = ax.plot(Xgps, Ygps[:,0], "+", mew=1, ms=6, color="C0", label="lat N", snap=False)#
    ax.set_ylabel(u"Latitude in $°$ N", color='C0')
    ax.tick_params('y', colors='C0')
    ax2 = ax.twinx()
    l2 = ax2.plot(Xgps, Ygps[:,1], "+", mew=1, ms=6, color="C1", label="lon E", snap=False)#
    ax2.set_ylabel(u"Longitude in $°$ E", color='C1')
    ax2.tick_params('y', colors='C1')
    ax.set_xlabel("$t$ in $s$")
    ax.set_xlim((-1,32))
    lns = l2+l1
    labs = [l.get_label() for l in lns]
    leg = ax.legend(lns, labs, bbox_to_anchor=(1, 0.5), loc='center right')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    pyplot.tight_layout()
    pyplot.show()
    
    pyplot.figure()
    pyplot.subplot(111)
    pyplot.title("Angle from gyroscope (integrated angular velocity)")
    intgyr = numpy.zeros(Ygyr.shape)
    intgyr[:,0] = integrate.cumtrapz(Ygyr[:,0], x=Xgyr[:,0], initial=0.0) * 180.0 / pi
    intgyr[:,1] = integrate.cumtrapz(Ygyr[:,1], x=Xgyr[:,0], initial=0.0) * 180.0 / pi
    intgyr[:,2] = integrate.cumtrapz(Ygyr[:,2], x=Xgyr[:,0], initial=0.0) * 180.0 / pi
    pyplot.plot(Xgyr, intgyr[:,0], label="yaw")# (clockwise facing upwards)")
    pyplot.plot(Xgyr, intgyr[:,1], label="pitch")# (clockwise facing leftwards)")
    pyplot.plot(Xgyr, intgyr[:,2], label="roll")# (clockwise facing backward)")
    pyplot.plot(Xgps[:-1], 
                [atan2(g[i+1,0]-g[i,0], g[i+1,1]-g[i,1]) * 180.0 / pi
                     for i in range(len(Xgps)-1)],
                 "--", label="$direction$ from gps")
    pyplot.xlabel("Time in $s$")
    pyplot.ylabel(u"Angular velocity in $°$")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    
    
    
