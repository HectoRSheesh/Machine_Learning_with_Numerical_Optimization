import numpy as np

def f(x):
    f = x[0]**2 - 2 * x[0] - 3 * x[1] * x[0] + 12 * x[1]#Function
    return f

def grad(x):
    gradf = [( 2 * x[0] - 3 * x[1] - 2) , ( 12 - 3 * x[0])] #Function's Gradiant Matrix Form
    return gradf

def hessian(x):
    hessianf = np.array([[2,-3],[-3,0]]) #Funciton's Hessien matrix Form
    return hessianf

x = [2,4]
i = 0
print("i:", i ," f(x):", f(x))
loop = True

while loop:
    i += 1

    x = x - 0.01 * np.array(grad(x))
    normGrad = np.linalg.norm(grad(x))

    print("i:", i ," f(x):", f(x), "//Grad//:", normGrad)

    if normGrad < 1e-8:
        loop = False

print("x*:", x)

H = hessian(x)
ozdeger, ozvektor = np.linalg.eigh(H)

if min(ozdeger) > 0:
    print("x noktası minimumdur")
elif max(ozdeger) < 0:
    print("x noktası maksimumdur")
else:
    print("x noktası semerdir")
