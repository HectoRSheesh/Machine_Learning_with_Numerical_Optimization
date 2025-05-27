#Created by Ozan Bali
# 03.19.2025
# Finding Root's Situation With Gradiant and Hessien Matrixes
"""
import numpy as np
def function(x):
    f = x[0]**2 - 2 * x[0] - 3 * x[1] * x[0] + 12 * x[1]#Function
    return f

def gradf(x):
    gradf = [( 2 * x[0] - 3 * x[1] - 2) , ( 12 - 3 * x[0])] #Function's Gradiant Matrix Form
    return gradf

def hessianf(x):
    hessianf = np.array([[2,-3],[-3,0]]) #Funciton's Hessien matrix Form
    return hessianf

x = [0,5]
i = 0 #Counter
print("i:",i," x:",x ," f(x):",function(x))
loop = True
while loop:
    i += 1
    x = x - 0.1 *  np.array(gradf(x))

    normGradf = np.linalg.norm(gradf(x))

    if normGradf > 1e10:
        print("Gradient norm is too large, stopping the loop.")
        loop = False
    print("i:",i," x:",x,"f(x):",function(x), "//Gradf//:",normGradf)
    if normGradf < 1e-6:
        loop = False

print("x*:",x)
Hessian = hessianf(x)
eigenvalue, eigenvector = np.linalg.eigh(Hessian)
print("Eigenvalues:", eigenvalue)
print("Eigenvectors:")
print(eigenvector)

if min(eigenvalue) > 0:
    print("x is minimum")
elif max(eigenvalue) < 0:
    print("x is maximum")
else:
    print("x is semer..")
"""
import numpy as np

# Sayıların bilimsel gösterimini bastır
np.set_printoptions(suppress=True)

def function(x):
    return x[0] ** 2 - 2 * x[0] - 3 * x[1] * x[0] + 12 * x[1]

def gradf(x):
    return np.array([2 * x[0] - 3 * x[1] - 2, 12 - 3 * x[0]])

def hessianf(x):
    return np.array([[2, -3], [-3, 0]])

x = np.array([0, 5], dtype=float)  # NumPy array kullanımı için uygun tanımlama
i = 0
print(f"i: {i}, x: {np.array2string(x, formatter={'float_kind': lambda x: f'{x:.6f}'})}, f(x): {function(x)}")

loop = True
while loop:
    i += 1
    x = x - 0.1 * gradf(x)
    normGradf = np.linalg.norm(gradf(x))

    if normGradf > 1e10:
        print("Gradient norm is too large, stopping the loop.")
        loop = False

    print(f"i: {i}, x: {np.array2string(x, formatter={'float_kind': lambda x: f'{x:.6f}'})}, f(x): {function(x)}, ||Gradf||: {normGradf:.6f}")

    if normGradf < 1e-6:
        loop = False

print(f"x*: {np.array2string(x, formatter={'float_kind': lambda x: f'{x:.6f}'})}")

Hessian = hessianf(x)
eigenvalue, eigenvector = np.linalg.eigh(Hessian)
print("Eigenvalues:", np.array2string(eigenvalue, formatter={'float_kind': lambda x: f'{x:.6f}'}))
print("Eigenvectors:")
print(np.array2string(eigenvector, formatter={'float_kind': lambda x: f'{x:.6f}'}))

if np.all(eigenvalue > 0):
    print("x* is a local minimum.")
elif np.all(eigenvalue < 0):
    print("x* is a local maximum.")
else:
    print("x* is a saddle point.")
