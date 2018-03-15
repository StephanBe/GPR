# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:14:50 2018

@author: Stephan
"""


"""
Integration XY-Fehler in Integration.py: X(Yacc[i,1]) vorwärts; Y(Yacc[i,2]) rechts

Length-Scale im RBF-Kernel ist, wie stark die Korrelation je Zeitschritt
bleibt (klein -> kaum Korrelation je weiter weg, groß -> stärkere Korrelation).

Constant-Kernel bestimmt glaube das sigma, also eventuell maximale Änderung
je Zeitschritt? (Vollbremsung, Unfall, Beschleunigung << Assymetrisch)

Beschleunigung aufintegrieren und Ort plotten (vs. GPS)

Accelerator-Pfeile in Fahrtrichtung rotieren.
(Fahrtrichtung = aktueller Ort - letzter Ort)

Über GPR Fusion der unterschiedlichen Größen recherchieren. Also Beschleunigung,
Rotation, Ort.

Kernel einstellen für ACC/GPS/whatever. Nachschauen, was die Parameter
bewirken. 



DONE (chronologisch):
Integration mit Kreisbeschleunigung getestet (numerisch instabil)

Werte, die fälschlicherweise auf einen Zeitstempel fallen, auffächern anhand
der Reihenfolge und des üblichen Abstands zwischen zwei Zeitstempeln.

Stefan Gumhold ne Mail schreiben, wegen der Zeitstempel

Verzeichnisse mit Zeitstempelduplikaten (VUFO) gezählt, Mailaustausch mit
Diana Hamelow (Verkehrsunfallforschung an der TU Dresden GmbH).

Video erklärt ganz gut GP in den ersten 10 min:
https://www.youtube.com/watch?v=BS4Wd5rwNwE

Ein paar Erklärungsplots gemacht zum Verständnis von GP ("Plot Multivariate
Gaussian"). 1D-GPR dabei selbst einmal implementiert.

Integration angefangen und einige Probleme gesehen.

Vektor-Rotation ins Gedächtnis gerufen (vectorRotation.py)
"""