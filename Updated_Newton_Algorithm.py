"""
Turkish version
import numpy as np
import math
import matplotlib.pyplot as plt

# Objective function
from ornekFonksiyon2 import f,hessian
from ornekFonksiyon2 import gradient as gradf

# Golden Section method to find step size
def GSmain(f, xk, pk):
    xbottom = 0
    xtop = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))

    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f(xk + x1 * pk)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f(xk + x2 * pk)

    for _ in range(N):
        if f1 > f2:
            xbottom = x1
            x1 = x2
            f1 = f2
            x2 = xtop - tau * (xtop - xbottom)
            f2 = f(xk + x2 * pk)
        else:
            xtop = x2
            x2 = x1
            f2 = f1
            x1 = xbottom + tau * (xtop - xbottom)
            f1 = f(xk + x1 * pk)

    result = 0.5 * (x1 + x2)
    return result

# Initialization
x = np.array([0,0])  # Starting point
x1 = [x[0]]  # Store x1 values for plotting
x2 = [x[1]]  # Store x2 values for plotting
Nmax = 10000  # Maximum iterations
eps1 = 1e-10  # Convergence tolerance for function value
eps2 = 1e-10  # Convergence tolerance for variable change
eps3 = 1e-10  # Convergence tolerance for gradient norm
k = 0  # Iteration counter

I = np.identity(2)
M = np.identity(2)
updatedx = np.array([1e10, 1e10])

C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

while not (C1 or C2 or C3 or C4):
    k += 1
    ozdeger , ozvektor = np.linalg.eigh(M)
    if np.min(ozdeger) > 0 :
        pk = -np.dot(M,gradf(x))
    else:
        mu = abs(np.min(ozdeger)) + 0.001
        pk = -np.dot((M + mu * I),gradf(x))
    sk = GSmain(f, x, pk)
    prevG = gradf(x)
    x = x + sk*pk
    x = np.array(x)
    currentG = gradf(x)
    y = (currentG - prevG).reshape(-1,1)
    Dx = (sk*pk).reshape(-1,1)

    A = np.dot(Dx,np.matrix.transpose(Dx)) / np.dot(np.matrix.transpose(Dx),y)
    B = np.dot(np.dot(M,y),np.matrix.transpose(np.dot(M,y))) / np.dot(np.matrix.transpose(y),np.dot(M,y))
    M = M + A - B

    print(
        f"k: {k}, sk: {round(sk, 6)}, x1: {round(x[0], 6)}, x2: {round(x[1], 6)}, f(x1,x2): {round(f(x), 6)}"
    )

    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3

    updatedx = 1 * x
    x1.append(x[0])
    x2.append(x[1])

if C1:
    print("Max iterasyon aşıldı")

if C2:
    print("fonksiyon değişmiyor")
if C3:
    print("değişkenler değişmiyor")
if C4:
    print("durağan noktaya gelindi.")

plt.plot (x1,x2)
plt.scatter(x1,x2,s =5,c = 'red')
plt.show()
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# Objective function, gradient, and Hessian imported from ornekFonksiyon2
from ornekFonksiyon2 import f, hessian
from ornekFonksiyon2 import gradient as gradf

# Golden Section method to find step size
def GSmain(f, xk, pk):
    xbottom = 0
    xtop = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))

    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f(xk + x1 * pk)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f(xk + x2 * pk)

    for _ in range(N):
        if f1 > f2:
            xbottom = x1
            x1 = x2
            f1 = f2
            x2 = xtop - tau * (xtop - xbottom)
            f2 = f(xk + x2 * pk)
        else:
            xtop = x2
            x2 = x1
            f2 = f1
            x1 = xbottom + tau * (xtop - xbottom)
            f1 = f(xk + x1 * pk)

    result = 0.5 * (x1 + x2)
    return result

# Initialization
x = np.array([0, 0])  # Starting point
x1 = [x[0]]  # Store x1 values for plotting
x2 = [x[1]]  # Store x2 values for plotting
Nmax = 10000  # Maximum number of iterations
eps1 = 1e-10  # Convergence tolerance for function value
eps2 = 1e-10  # Convergence tolerance for variable change
eps3 = 1e-10  # Convergence tolerance for gradient norm
k = 0  # Iteration counter

I = np.identity(2)
M = np.identity(2)
updatedx = np.array([1e10, 1e10])

C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

while not (C1 or C2 or C3 or C4):
    k += 1
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    if np.min(eigenvalues) > 0:
        pk = -np.dot(M, gradf(x))
    else:
        mu = abs(np.min(eigenvalues)) + 0.001
        pk = -np.dot((M + mu * I), gradf(x))
    sk = GSmain(f, x, pk)
    prevG = gradf(x)
    x = x + sk * pk
    x = np.array(x)
    currentG = gradf(x)
    y = (currentG - prevG).reshape(-1, 1)
    Dx = (sk * pk).reshape(-1, 1)

    A = np.dot(Dx, np.transpose(Dx)) / np.dot(np.transpose(Dx), y)
    B = np.dot(np.dot(M, y), np.transpose(np.dot(M, y))) / np.dot(np.transpose(y), np.dot(M, y))
    M = M + A - B

    # Print iteration details
    print(
        f"Iteration: {k}, Step size: {round(sk, 6)}, x1: {round(x[0], 6)}, x2: {round(x[1], 6)}, f(x1, x2): {round(f(x), 6)}"
    )

    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3

    updatedx = x.copy()
    x1.append(x[0])
    x2.append(x[1])

if C1:
    print("Maximum number of iterations reached.")

if C2:
    print("Function value is not changing.")
if C3:
    print("Variables are not changing.")
if C4:
    print("Stationary point reached.")

# Improved visualization
plt.figure(figsize=(10, 6))
plt.plot(x1, x2, label="Optimization Path", color="blue", linewidth=1.5)
plt.scatter(x1, x2, s=20, color="red", label="Steps")
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)
plt.title("Optimization Path Using Modified Newton Algorithm", fontsize=14)
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

