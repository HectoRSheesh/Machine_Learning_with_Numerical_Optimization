import numpy as np
import math
from dataset2 import ti, yi
#------------------------------------------------------------------------------
def exponentialIO(t,x):
    yhat = []
    for ti in t:
        toplam = x[0]*math.exp(x[1]*ti)
        yhat.append(toplam)
    return yhat
#------------------------------------------------------------------------------
def error(xk, ti, yi):
    yhat = exponentialIO(ti, xk)
    return np.array(yi) - np.array(yhat)
#------------------------------------------------------------------------------
def findJacobian(traininginput, x):
    numofdata = len(traininginput)
    J = np.matrix(np.zeros((numofdata,2)))
    for i in range(0, numofdata):
        J[i,0] = -math.exp(x[1]*traininginput[i])
        J[i,1] = -x[0]*traininginput[i]*math.exp(x[1]*traininginput[i])
    return J
#------------------------------------------------------------------------------
trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
#------------------------------------------------------------------------------
x1 = [np.random.random()-0.5]
x2 = [np.random.random()- 0.5]
xk = np.array([x1[0], x2[0]])

k = 0; C1 = True; C2 = True; C3= True; C4 = True; fvalidationBest = 1e99; kbest = 0

ek = error(xk, traininginput, trainingoutput)
ftraining = sum(ek**2)
FTRA = [ftraining]
evalidation = error(xk, validationinput, validationoutput)
fvalidation = sum(evalidation**2)
FVAL = [fvalidation]
ITERATİON = [k]
print('k:', k, 'x1:', format(xk[0],'f'), 'x2:', format(xk[1],'f'), 'f', format(ftraining,'f'))
mu = 1; muscal= 10; I = np.identity(2)
while C1 & C2 & C3 & C4:
    ek = error(xk, traininginput,trainingoutput)
    Jk = findJacobian(traininginput, xk)
    gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])
    Hk = 2*Jk.transpose().dot(Jk) + 1e-8 * I
    ftraining = sum(ek**2)
    sk = 1
    loop = True
    while loop:
        zk = -np.linalg.inv(Hk + mu*I).dot(gk)
        zk = np.array(zk.tolist()[0])
        ez = error(xk + sk*zk, traininginput, trainingoutput)
        fz = sum(ez**2)
        if fz < ftraining:
            pk = 1*zk
            mu = mu/muscal
            k += 1
            xk = xk +sk*pk
            x1.append(xk[0])
            x2.append(xk[1])
            loop = False
            print('k:', k, 'x1:', format(xk[0],'f'), 'x2:', format(xk[1],'f'), 'f', format(ftraining,'f'))
        else:
            mu = mu*muscal
            if mu > mumax:
                loop = False
                C2 = False
    evalidation = error(xk, validationinput, validationoutput)
    fvalidation = sum(evalidation**2)
    if fvalidation < fvalidationBest:
        fvalidationBest = 1*fvalidation
        xkbest = 1*xk
        kbest = k
    FTRA.append(ftraining)
    FVAL.append(fvalidation)
    ITERATİON.append(k)
    #-----------------------
    C1 = k < MaxIter
    C2 = epsilon1 < abs(ftraining - fz)
    C3 = epsilon2 < np.linalg.norm(sk*pk)
    C4 = epsilon3 < np.linalg.norm(gk)
    #-----------------------
print('xkbest: ', xkbest)
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(ITERATİON, FTRA, color = 'green', linestyle = 'solid', linewidth = 1)
plt.plot(ITERATİON, FVAL, color = 'red', linestyle = 'solid', linewidth = 1)
plt.axvline(x = kbest, color = 'b', linewidth = 1, linestyle = 'dashed')
plt.xlabel('iterasyon')
plt.ylabel('performanslar')
plt.title(' performanslar', fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
plt.legend(['training', 'validation'])
plt.show()
