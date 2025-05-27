import numpy as np
import math
from dataset2 import ti,yi
from exponentialModel3 import ftraining
from polinomIO import fvalidation, plotresult
k =0


#-------------------------------------------------------------------------------------
def exponentialIO(t,x): # t -> inputlar, x  -> parametreler
    yhat = []  #tüm çıkışları tanımlamak için kullancağız
    for ti in t:
        toplam = x[0]*math.exp(x[1]*ti)
        yhat.append(toplam)
    return yhat
#-------------------------------------------------------------------------------------

def error(xk,ti, yi):
    yhat = exponentialIO(ti, xk)
    return np.array(yi) - np.array(yhat)  #Gerçek çıktı - Üretilen çıktı
#-------------------------------------------------------------------------------------

def findJacobian(traininginput,x):
    numofdata = len(traininginput)
    J = np.matrix(np.zeros((numofdata,2)))
    for i in range(0, numofdata):
        J[i,0] = -math.exp(x[1]*traininginput[i]) # Jacobian 1. sutün
        J[i,1] = -x[0]*traininginput[i]*math.exp(x[1]*traininginput[i]) # Jacobian 2. sutün
    return J

#-------------------------------------------------------------------------------------

trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validatoninput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
NODEmax = int((len(traininginput)-1)/3)

NODE = []; FV = []; globalbest = 1e10; Sbest = 0;

for S in range(2,NODEmax):
    xk = np.random.random(3*S+1) - 0.5
    K = 0; C1 = True; C2 = True; C3 = True; C4 = True; fvalidationBest = 1e99; kbest = 0
    ek = error(xk,traininginput,trainingoutput)
    ftraining = sum(ek**2)
    FTRA = [ftraining]
    evalidation = error(xk,validatoninput,validationoutput)
    fvalidation = sum(evalidation**2)
    FVAL = [fvalidation]
    ITERATION = [k]

    mu = 1; muscal = 10; I = np.identity(3*S+1)
    while C1 & C2 & C3 & C4:
        ek = error(xk,traininginput,trainingoutput)
        Jk = findJacobian(traininginput,xk)
        gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])
        Hk = 2*Jk.transpose().dot(Jk) + 1e-8*I
        ftraining = sum(ek**2)
        sk = 1
        loop = True
        while loop:
            zk = -np.linalg.inv(Hk + mu * I).dot(gk)  # Aday yön
            zk = np.array(zk.tolist()[0])
            ez = error(xk + sk * zk, traininginput, trainingoutput)
            fz = sum(ez ** 2)
            if fz < ftraining:
                pk = 1 * zk
                mu = mu / muscal
                k += 1
                xk = xk + sk * pk
                loop = False
            else:
                mu = mu * muscal
                if mu > mumax:
                    loop = False
                    C2 = False

        evalidation = error(xk,validationinput,validationoutput)
        fvalidation = sum(evalidation**2)
        if fvalidation<fvalidationBest:
            fvaldationBest = 1*fvalidation
            xkbest = 1*xk
            kbest = k
        FTRA.append(ftraining)
        FVAL.append(fvalidation)
        ITERATION.append(k)
        C1 = k < MaxIter
        C2 = epsilon1 < abs(ftraining - fz)
        C3 = epsilon2 < np.linalg.norm(sk * pk)
        C4 = epsilon3 < np.linalg.norm(gk)

    plotresult(ti,yi,xkbest)
    NODE.append(S)
    FV.append(fvalidationBest)
    if fvalidationBest<globalbest:
        globalbest = 1*fvalidationBest
        Sbest = S
    print('Düğüm Sayisi:',S,'FVALBEST:',fvaldationBest,'GlobalFvalBest:',globalbest)

import matplotlib.pyplot as plt

plt.bar(NODE,FV,color='orange',width=0.4,linestyle = 'solid',linewidth=1)
plt.bar(Sbest,globalbest,color='blue',width=0.4,linestyle = 'solid',linewidth=1)
plt.xlabel('RBFSayisi')
plt.ylabel('Validation performansı')
plt.title('RBF Modeli Validasyon Performansları',fontstyle='italic')
plt.grid(color='green',linestyle='--',linewidth=0.1)
plt.show()

