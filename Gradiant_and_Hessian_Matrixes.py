#Created by Ozan Bali
# 03.19.2025
# Finding Root's Situation With Gradiant and Hessien Matries

import numpy as np
def function(x):
    f = (x[0] - 1)**2 + (x[1] - 1)**2 - x[0]*x[1] #Function
    return f

def gradf(x):
    gradf = [(2*(x[0] - 1)-x[1]), (2*(x[1] - 1)-x[0])] #Function's Gradiant Matrix Form
    return gradf

def hessianf(x):
    hessianf = np.array([[2,-1],[-1,2]]) #Funciton's Hessien matrix Form
    return hessianf

x = [1,1]
i = 0 #Counter
print("i:",i," f(x):",function(x))
loop = True
while loop:
    i += 1
    x = x - 0.1 *  np.array(gradf(x))
    normGradf = np.linalg.norm(gradf(x))
    print("i:",i," f(x):",function(x), "//Gradf//:",normGradf)

    if normGradf < 1e-6:
        loop = False

print("x*:",x)
Hessian = hessianf(x)
eigenvalue, eigenvector = np.linalg.eigh(Hessian)

if min(eigenvalue) > 0:
    print("x is minimum")
elif max(eigenvalue) < 0:
    print("x is maximum")
else:
    print("x is semer..")