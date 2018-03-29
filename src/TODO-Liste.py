# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:14:50 2018

@author: Stephan
"""


"""
In "Plot Multivariate Gaussian.py":
    sigma_star = np.diagonal(sigma_star).reshape(-1,1)
Ist das richtig? Die Formeln aus den Quellen nehmen für sigma_star (oder
K_posterior) immer K_star_star, was ja eine NxN Matrix ist (N Anzahl der
vorherzusagenden Elemente).

Length-Scale im RBF-Kernel ist, wie stark die Korrelation je Zeitschritt
bleibt (klein -> kaum Korrelation je weiter weg, groß -> stärkere Korrelation).

Constant-Kernel bestimmt glaube das sigma, also eventuell maximale Änderung
je Zeitschritt? (Vollbremsung, Unfall, Beschleunigung << Assymetrisch?!)

Beschleunigung aufintegrieren und Ort plotten (vs. GPS)

Accelerator-Pfeile in Fahrtrichtung rotieren.
(Fahrtrichtung = aktueller Ort - letzter Ort)

Über GPR Fusion der unterschiedlichen Größen recherchieren. Also Beschleunigung,
Rotation, Ort.

Kernel einstellen für ACC/GPS/whatever. Nachschauen, was die Parameter
bewirken. 

Orientierung des Autos in der Welt schätzen
- integrierte Gyro-Daten mit Beschleunigungsdaten (Suche nach Erdbeschleunigung)
korrigieren
- z.B. mit Madgwick's algirithm / Mahony's algorithm / Kalman Filter

scipy.integrate.ode anschauen für Integration

DONE (chronologisch):
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

Integration mit Kreisbeschleunigung getestet (numerisch instabil)

Problem bei Integration ist, dass durch die Schätzung der aktuellen Richtung
durch die aktuelle und die vorangegangene Position in Kurven immer ein Drift
nach außen geschieht. Besser: Approximation zusätzlich mit der zukünftigen
Position (approximiert). Und Bug mit den epsilon-Werten behoben, dadurch ist
nun auch eine höhere Auflösung zuträglich für die Genauigkeit der Integration.

GPS GP Kernel gefunden, sodass GPS auch nichtstationär sein kann:
- Linearerer Kernel: https://www.cs.toronto.edu/~duvenaud/cookbook/
- bzw. Polynomial mit p=1: https://de.wikipedia.org/wiki/Gau%C3%9F-Prozess
- oder auch Dot Product Kernel.
In Plot Multivariate Gaussian implementiert. In scikit-learn kann man dafür
den DotProduct kernel nehmen.

Integration XY-Fehler in Integration.py:
X(Yacc[i,1]) vorwärts; Y(Yacc[i,2]) links
Das lag vielmehr an dem Offset der Beschleunigungsdaten, wodurch sich das
Auto virtuell langsam im Stand drehte. Mit den Daten von
Erprobung\\Fahrsicherheitstraining\\Ausweichen Touran
bei stehendem Auto (bis etwa 120ster Eintrag) und der Anweisung
Yacc[:,1] = Yacc[:,1]-mean(Yacc[0:120,1])
Yacc[:,2] = Yacc[:,2]-mean(Yacc[0:120,2])
erstmal kaschiert.
Das muss dann in einer finalen Lösung ordentlicher gemacht werden.
"""