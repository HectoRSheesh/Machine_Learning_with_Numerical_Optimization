#Created by Ozan Bali
# 03.19.2025
# Finding Root's Situation With Gradiant and Hessien Matrixes

import numpy as np
def function(x):
    f = (x[0]**3) / 3 + (x[0]**2) / 2 + 2* (x[0]*x[1])  + (x[1]**2) / 2 - x[1] + 9#Function
    return f

def gradf(x):
    gradf = [(x[0]**2 + x[0] + 2*x[1]), (2*x[0] + x[1] - 1)] #Function's Gradiant Matrix Form
    return gradf

def hessianf(x):
    hessianf = np.array([[5,2],[2,1]]) #Funciton's Hessien matrix Form
    return hessianf

x = [1,1]
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