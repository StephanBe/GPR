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
from math import cos
import platform


if platform.system() == 'Windows':
    vufo_data = 'data\\Erprobung\\Fahrsicherheitstraining\\Ausweichen Touran'
else:
    vufo_data = '../data/Erprobung/Fahrsicherheitstraining/Ausweichen Touran'

"""liest VUFO-CSV-Daten ein"""
def read(filename):
    Y = []
    X = []
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            X.append(datetime.strptime(row[0]+' '+row[1], '%d.%m.%Y %H:%M:%S,%f'))
            Y.append([float(x.replace(',', '.')) for x in row[2:]])
    #X = numpy.asarray(X)
    """Normierung auf den Start und Konvertierung in Sekunden"""
    X = numpy.atleast_2d([(x-min(X)).total_seconds() for x in X]).T
    Y = numpy.asarray(Y)
    return [X, Y]

"""transformiert Höhen- und Breitengrad in Meter"""
def latlonToMeter(latlon):
    #Mittlere Höhe und Breite für Krümmung
    l = numpy.mean(latlon[:,0])
    #Meter pro Breitengrad
    m_per_deg_lat = 111132.92
    m_per_deg_lat -= 559.82 * cos(2.0 * l)
    m_per_deg_lat += 1.175 * cos(4.0 * l)
    m_per_deg_lat -= 0.0023 * cos(6.0 * l)
    #Neter pro Längengrad
    m_per_deg_lon = 111412.84 * cos(l)
    m_per_deg_lon -= 93.5 * cos(3.0 * l)
    m_per_deg_lon += 0.118 * cos(5.0 * l)
    #Normieren und umrechnen in Meter
    norm = []
    for x in latlon:
        norm.append([(x[0]-latlon[0,0])*m_per_deg_lat,
                     (x[1]-latlon[0,1])*m_per_deg_lon])
    return numpy.asarray(norm)

"""zieht Mittelwert der Daten ab"""   
def normalizeAcc(Yacc):
    y = [[Y[0]-numpy.mean(Yacc[:,0]),
          Y[1]-numpy.mean(Yacc[:,1]),
          Y[2]-numpy.mean(Yacc[:,2])]
          for Y in Yacc]
    return  numpy.asarray(y)

"""springe in das gegebene Verzeichnis"""
prevdir=os.getcwd()
os.chdir(vufo_data)
"""Beschleunigung einlesen"""
Xacc, Yacc = read('AccelData.txt')
"""GPS-Daten einlesen"""
Xgps, Ygps = read('GPSData.txt')
"""Gyro-Daten einlesen"""
Xgyr, Ygyr = read('GyroData.txt')
os.chdir(prevdir)

"""Korrigiere Zeitstempelfehler"""
Xacc_corrected = numpy.copy(Xacc)
delta = Xacc[1:]-Xacc[:-1]
normal = numpy.median(delta)
i = 0
while i < len(Xacc)-1:
    if Xacc_corrected[i+1]-Xacc_corrected[i] < 0.5 * normal:
        """Korrigiere Anfang eines Fehlers"""
        Xacc_corrected[i+1] = Xacc_corrected[i] + normal
    i += 1

Xacc_corrected2 = numpy.copy(Xacc)
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


if __name__ == "__main__":
    pyplot.figure()
    pyplot.subplot(231)
    pyplot.plot(Xacc, label="original")
    pyplot.plot(Xacc_corrected2, "--", label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected, "--", label="corrected by keeping a normal time frame")
    pyplot.title("Time Stamp\nOf Each Data Entry")
    pyplot.xlabel("$i$th entry")
    pyplot.ylabel("$t$ in $s$")
#==============================================================================
#     pyplot.legend()
#==============================================================================
    
    pyplot.subplot(232)
    pyplot.plot(Xacc[1:] - Xacc[:-1], label="original")
    pyplot.plot(Xacc_corrected2[1:] - Xacc_corrected2[:-1], "--", label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected[1:] - Xacc_corrected[:-1], "--", label="corrected by keeping a normal time frame")
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
    pyplot.plot(Xacc_corrected2-medianDelta, "--", label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected-medianDelta, "--", label="corrected by keeping a normal time frame")
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
    pyplot.plot(Xacc, Yacc[:,1], label="original")
    pyplot.plot(Xacc_corrected2, Yacc[:,1], "--", label="corrected by evenly distributing time")
    pyplot.plot(Xacc_corrected, Yacc[:,1], "--", label="corrected by keeping a normal time frame")
    pyplot.xlabel("$t$ in $s$")
    pyplot.ylabel("$a$ in $m*s^{-1}$")
    pyplot.legend()
    pyplot.show()
    
    pyplot.figure()
    pyplot.subplot(121)
    pyplot.title("Car Position")
    g = latlonToMeter(Ygps)
    cb = pyplot.scatter(g[:,1], g[:,0], c=Xgps.flatten())
    pyplot.colorbar(cb, label="$t$ in $s$")
    pyplot.xlabel("longitude in $m$")
    pyplot.ylabel("latitude in $m$")
    pyplot.subplot(122)
    pyplot.title("Car Acceleration")
    pyplot.plot(Xacc, Yacc[:,0], label="$a_{down}$")
    pyplot.plot(Xacc, Yacc[:,1], label="$a_{left}$")
    pyplot.plot(Xacc, Yacc[:,2], label="$a_{back}$")
    pyplot.xlabel("$t$ in $s$")
    pyplot.ylabel("$a$ in $m*s^{-1}$")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    