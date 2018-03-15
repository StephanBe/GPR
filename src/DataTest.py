# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:19:11 2018

@author: Stephan
"""

"""Daten überprüfen"""

import numpy
import csv
from datetime import datetime
import os
from os.path import join
#from matplotlib import pyplot

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

def hasDuplicates(x):
    where = next((i for i in range(len(x)-1) if x[i] - x[i-1] == 0), -1)
    f = x.flatten()
    return len(f) != len(set(f)), where

prevdir=os.getcwd()
#os.chdir('data')

n=0
d=0
with open('Liste_der_Dopplungen.txt', 'w') as f:
    for root, subdirs, files in os.walk('data'):
        if 'AccelData.txt' not in files or\
             'GPSData.txt' not in files or\
             'GyroData.txt' not in files:
                 continue
        #print('--\nroot = ' + root)
        #print('subdirs = [',end='');print(*subdirs, sep=', ', end='');print(']')
        #print('files = [',end='');print(*files, sep=', ', end='');print(']')
        duplicate = False
        if 'AccelData.txt' in files:
            x, y = read(join(root, 'AccelData.txt'))
            hasD, where = hasDuplicates(x)
            if(hasD):
                duplicate = True
                f.write('First duplicate at ' + str(where) + ': ' + root + '\\AccelData.txt\n')
        if 'GPSData.txt' in files:
            x, y = read(join(root, 'AccelData.txt'))
            hasD, where = hasDuplicates(x)
            if(hasD):
                duplicate = True
                f.write('First duplicate at ' + str(where) + ': ' + root + '\\GPSData.txt')
        if 'GyroData.txt' in files:
            x, y = read(join(root, 'AccelData.txt'))
            hasD, where = hasDuplicates(x)
            if(hasD):
                duplicate = True
                f.write('First duplicate at ' + str(where) + ': ' + root + '\\GyroData.txt')
        if duplicate:
            d= d+1
        else:
            n = n+1
            #print(root)
    f.write('\nNormal: {} Duplikate: {}'.format(n, d))
    
"""
#Beschleunigung einlesen
Xacc, Yacc = read('AccelData.txt')

#GPS-Daten einlesen
Xgps, Ygps = read('GPSData.txt')

#Gyro-Daten einlesen
Xgyr, Ygyr = read('GyroData.txt')
"""

os.chdir(prevdir)