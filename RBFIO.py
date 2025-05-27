import numpy as np
import math
from dataset2 import ti, yi
import matplotlib.pyplot as plt

def gaussianfunction(t, c, s):
    h = math.exp(-((t - c)**2) / (s**2))
    return h

def RBFIO(t, x, c, s):
    yhat = []
    for ti_val in t:
        toplam = 0
        for i in range(len(x)):
            toplam += x[i] * gaussianfunction(ti_val, c[i], s[i])
        yhat.append(toplam)
    return yhat

def findxcs(ti, yi, RBFsayisi):
    lengthofsegment = (max(ti) - min(ti)) / RBFsayisi
    s = [lengthofsegment for _ in range(RBFsayisi)]
    c = [min(ti) + lengthofsegment/2 + lengthofsegment * i for i in range(RBFsayisi)]
    numofdata = len(ti)
    J = np.zeros((numofdata, RBFsayisi))
    for i in range(numofdata):
        for j in range(RBFsayisi):
            J[i, j] = -gaussianfunction(ti[i], c[j], s[j])
    A = np.linalg.inv(J.T.dot(J))
    B = J.T.dot(yi)
    x = -A.dot(B)
    return x, c, s

def plotresult(ti, yi, x, c, s, fvalidation):
    T = np.arange(min(ti), max(ti), 0.1)
    yhat = RBFIO(T, x, c, s)
    plt.scatter(ti, yi, color='darkred', marker='x')
    plt.plot(T, yhat, color='green', linestyle='solid', linewidth=1)
    plt.xlabel('ti')
    plt.ylabel('yi')
    plt.title(f"{len(x)} düğümlü RBF modeli | FV: {fvalidation}",
              fontdict={'fontsize': 12, 'fontstyle': 'italic'})
    plt.grid(color='green', linestyle='--', linewidth=0.1)
    plt.legend(['RBF modeli', 'gerçek veri'])
    plt.show()

trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

RBF = []
FV = []
for RBFsayisi in range(1, 9):
    x, c, s = findxcs(traininginput, trainingoutput, RBFsayisi)
    yhat = RBFIO(validationinput, x, c, s)
    e = np.array(validationoutput) - np.array(yhat)
    fvalidation = sum(e**2)
    RBF.append(RBFsayisi)
    FV.append(fvalidation)
    print(RBFsayisi, fvalidation)
    plotresult(ti, yi, x, c, s, fvalidation)

plt.bar(RBF, FV, color='darkred')
plt.xlabel('RBF sayisi')
plt.ylabel('Validation Performansı')
plt.title('RBF Modeli', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
plt.grid(color='green', linestyle='--', linewidth=0.1)
plt.show()
