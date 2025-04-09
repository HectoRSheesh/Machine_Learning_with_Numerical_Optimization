import math
import numpy as np
import pkce
from sympy import true

from ornekFonksiyon2 import f, hessian, jacobian, error
from ornekFonksiyon2 import gradient as gradf


def GSmain(f, xk, pk):
    a = 0
    b = 1
    tol = 1e-5
    alpha = (math.sqrt(5) - 1) / 2

    x1 = b - alpha * (b - a)
    x2 = a + alpha * (b - a)
    f1 = f(xk + x1 * pk)
    f2 = f(xk + x2 * pk)

    while (b - a) > tol:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + alpha * (b - a)
            f2 = f(xk + x2 * pk)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - alpha * (b - a)
            f1 = f(xk + x1 * pk)

    return (a + b) / 2


# ADIM 1 --------

x = np.array([0.0, 0.0])
X1 = [x[0]]
X2 = [x[1]]
muk = 1
mumin = 1e-20
mumax = 1e+20
muscale = 10
Nmax = 10000
eps1 = 1e-10
eps2 = 1e-10
eps3 = 1e-10
k = 0

I = np.identity(2)

updatedx = np.array([1e10, 1e10])
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

# ADIM 2 -------

while not (C1 or C2 or C3 or C4):
    k += 1
    J = jacobian(x)
    e = error(x)
    A = np.dot(np.matrix.transpose(J), J)
    B = np.dot(np.matrix.transpose(J), e.reshape(-1, 1))
    loop = True
    while loop:
        zk = -np.dot(np.linalg.inv(A + muk * I), B)
        zk = np.array(zk).reshape(-1, )
        if f(x + zk) < f(x):
            sk = GSmain(f, x, zk)
            x = x + sk * zk
            x = np.array(x)
            muk = muk / muscale
            loop = False
        else:
            muk = muk * muscale
            if mumax < muk:
                loop = False
    print("k: ", k, "sk: ", round(sk, 4), "x1: ", round(x[0], 4), "x2: ", round(x[1], 4),
          "f: ", round(f(x), 4), "||gradf||: ", round(np.linalg.norm(gradf(x)), 4))
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    updatedx = 1 * x
    X1.append(x[0])
    X2.append(x[1])

if C1:
    print("...maksimum iterasyon sayısına ulaşıldı")
if C2:
    print("...fonksiyon değişmiyor")
if C3:
    print("...değişkenler değişmiyor")
if C4:
    print("...durağan noktaya gelindi")

import matplotlib.pyplot as plt

plt.plot(X1, X2)
plt.scatter(X1, X2, s=5, c='red')
plt.show()
