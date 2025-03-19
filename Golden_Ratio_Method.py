# Created by Ozan Bali
# 03.19.2025
# Golden Ratio Method

import math
def function(x):
    f = (x-1)**2 * (x-2) * (x-3) #Function
    return f
def function_derivative(x):
    f = 2 * (x-1) * (x-2) * (x-3) + (x-1)**2 * (2*x-5)
    return f


x_bottom = 2 # Bottom value
x_top = 3 # Top value

dx = 0.0000001 # Exchange ratio (You can change)

alpha = (1 + math.sqrt(5)) / 2
tau = 1 - 1 / alpha
epsilon = dx / (x_top - x_bottom)

N = round(-2.078 * math.log(epsilon)) # Iteration count
k = 0 #Counter

x1 = x_bottom + tau * (x_top - x_bottom)
x2 = x_top - tau * (x_top - x_bottom)
f1 = function(x1)
f2 = function(x2)

for k in range(N):
    if f1 > f2:
        x_bottom = 1 * x1
        x1 = 1 * x2
        f1 = 1 * f2
        x2 = x_top - tau * (x_top - x_bottom)
        f2 = function(x2)

    else:
        x_top = 1 * x2
        x2 = 1 * x1
        f2 = 1 * f1
        x1 = x_bottom + tau * (x_top - x_bottom)
        f1 = function(x1)

result = 0.5 * (x1 + x2) #Result state
print("x_root:",round(result,5)," f: ",round(function(result),5)," f1:",round(function_derivative(result),5))