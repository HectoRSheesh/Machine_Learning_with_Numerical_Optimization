import numpy as np
import math
import matplotlib.pyplot as plt

# Objective function
def f(x):
    return 3 + (x[0] - 1.5 * x[1])**2 + (x[1] - 2)**2

# Gradient of the function
def gradf(x):
    return np.array([2 * (x[0] - 1.5 * x[1]), -3 * (x[0] - 1.5 * x[1]) + 2 * (x[1] - 2)])

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
x = np.array([-4.5,-3.5])  # Starting point
x1 = [x[0]]  # List to store x1 values for plotting
x2 = [x[1]]  # List to store x2 values for plotting
Nmax = 10000  # Maximum iterations
eps1 = 1e-10  # Convergence tolerance for function value
eps2 = 1e-10  # Convergence tolerance for variable change
eps3 = 1e-10  # Convergence tolerance for gradient norm
k = 0  # Iteration counter

updatedx = np.copy(x)  # Updated x initialization
while k < Nmax:
    pk = -gradf(x)  # Steepest descent direction
    sk = GSmain(f, x, pk)  # Step size determination using golden section search
    updatedx = x + sk * pk  # Update x

    # Convergence conditions
    if abs(f(updatedx) - f(x)) < eps1:
        print("Function value convergence reached.")
        break
    if np.linalg.norm(updatedx - x) < eps2:
        print("Variable change convergence reached.")
        break
    if np.linalg.norm(gradf(updatedx)) < eps3:
        print("Gradient norm convergence reached.")
        break

    # Update variables for the next iteration
    x = updatedx
    x1.append(x[0])
    x2.append(x[1])
    k += 1
    print(f"Iteration {k}: x = {x}, f(x) = {f(x)}, Gradient Norm = {np.linalg.norm(gradf(x))}")

if k == Nmax:
    print("Maximum iteration limit reached.")
print(f"Final x_root: {x}")

# Visualization of optimization path
plt.plot(x1, x2, label="Optimization Path")
plt.scatter(x1, x2, s=5, c='red', label="Steps")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Steepest Descent Path")
plt.legend()
plt.show()
