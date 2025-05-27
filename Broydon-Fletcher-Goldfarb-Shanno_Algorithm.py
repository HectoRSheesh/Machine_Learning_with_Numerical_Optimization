import numpy as np
import math
import matplotlib.pyplot as plt

from ornekFonksiyon2 import f
from ornekFonksiyon2 import gradient as gradf


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

    return result # Return the midpoint of the final interval as the optimal step size


# --- STEP 1: Initialization ---

# Set the starting point
x = np.array([-1.5, -1.5])

# Define the maximum number of iterations
Nmax = 10000

# Termination criteria thresholds:
eps1 = 1e-10  # Threshold for change in function value
eps2 = 1e-10  # Threshold for change in the variable x
eps3 = 1e-10  # Threshold for the gradient norm (stationary point)

k = 0  # Iteration counter

# Set the initial metric matrix (M0) as the identity matrix
I = np.identity(2)
M = np.identity(2)
updatedx = np.array([1e10, 1e10])
# Lists to record the history of iterations for plotting
x1 = [x[0]]
x2 = [x[1]]

C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x ) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3
# --- STEP 2: Iterative Process ---

while not (C1 or C2 or C3 or C4):
    k += 1
    ozdeger,ozvektor = np.linalg.eigh(M)
    if np.min(ozdeger) > 0:
        pk = -np.dot(np.linalg.inv(M), gradf(x))
    else:
        mu = abs(np.min(ozdeger)) + 0.001
        pk = -np.dot((np.linalg.inv(M + mu*I)),gradf(x))

    sk = GSmain(f, x, pk)
    prevG = gradf(x).reshape(-1,1)
    x = x + sk * pk
    x = np.array(x)
    currentG = gradf(x).reshape(-1,1)
    y = (currentG - prevG)
    pk = pk.reshape(-1,1)
    Dx = (sk*pk)

    A = np.dot(y,np.matrix.transpose(y)) / np.dot(np.matrix.transpose(y), Dx)
    B = np.dot(np.dot(M,Dx),np.dot(np.matrix.transpose(Dx),M)) / np.dot(np.matrix.transpose(Dx), np.dot(M,Dx))

    M = M + A - B

    k += 1  # Increment iteration counter
    print("Iteration:", k, " Step size:", sk, " x:", np.round(x, 4), " f(x):", np.round(f(x), 4))
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    updatedx = 1*x
    x1.append(x[0])
    x2.append(x[1])
if C1:
    print("maksimum iterasyona ulaşıldı")
if C2:
    print("fonksiyon değişmiyor")
if C3:
    print("değişkenler değişmiyor")
if C4:
    print("durağan noktaya gelindi")


# --- STEP 3: Plot the Iteration History ---
plt.figure(figsize=(8, 6))
plt.plot(x1, x2, marker='o', linestyle='-', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Iteration History')
plt.grid(True)
plt.show()
