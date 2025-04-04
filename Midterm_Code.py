import numpy as np

def f(x):
    f = (x[0] - 1.5)**2 + (x[1] - 2.5)**2
    return f

def grad(x):
    grad = [2*(x[0] - 1.5),2*(x[1] - 2.5)]
    return grad

def hessian(x):
    hessian = np.array([[5,2],[2,1]])
    return hessian

x = [4,3]
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
