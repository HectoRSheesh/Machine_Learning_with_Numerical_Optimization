import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    f = 3 + (x[0]-1.5 * x[1])**2 + (x[1] - 2)**2
    return f

def gradf(x):
    gradf = np.array(2*(x[0]-1.5*x[1]), -3 * (x[0]-1.5*x[1])+ 2* (x[1]-2)])
    return gradf

def GSmain (f,xk,pk):
    xalt = 0
    xüst = 1
    dx =  0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1/alpha
    epsilon = dx / (xüst - xalt)
    N = round(-2.078 * math.log(epsilon))

    k = 0
    x1 = xalt + tau * (xüst - xalt)
    f1 = f(xk + x1 * pk)
    x2 = xüst - tau * (xüst - xalt)
    f2 = f(xk + x2 * pk)
    print("k:",k," x1:",round(x1,5)," x2:",round(x2,5)," f1:",round(f1,5)," f2:",round(f2,5))

    for k in range(0,N):
        if f1 > f2:
            xalt = 1 * x1
            x1 = 1 * x2
            f1 = 1 * f2
            x2 = xüst - tau * (xüst - xalt)
            f2 = f(xk + x2 * pk)
        else:
            xüst = 1 * x2
            x2 = 1 * x1
            f2 = 1 * f1
            x1 = xalt + tau * (xüst - xalt)
            f1 = f(xk + x1 * pk)
x = 0.5 * (x1 + x2)
return x

x = np.array([-5.4,1.7])
x1 = [x[0]]
x2 = [x[1]]
Nmax = 10000
eps1 = 1e-10
eps2 = 1e-10
eps3 = 1e-10
k = 0

updatedx = np.array([1e10,1e10])
C1 = Nmax < k
C2 = abs(f(updatedx)-f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

while not (C1 or C2 or C3 or C4):
    k += 1
    pk = -gradf(x)
    sk = GSmain(f,x,pk)
    x = x + sk*pk
    print("k:",k," sk:",sk," x:",x," x1:",x1," x2:",x2," f:",round(f(x),4),"gradf:",round(gradf(x),4))
    C1 = Nmax < k
    C2 = abs(gradf(x)-f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3

    updatedx = 1 * x
    x1.append(x[0])
    x2.append(x[1])
if C1:
    print("...max iterasyon sayısı aşıldı")
if C2:
    print("fonksiyon değişmiyor")
if C3:
    print("değişkenler değişmiyor")
if C4:
    print("durağan noktaya gelindi")






