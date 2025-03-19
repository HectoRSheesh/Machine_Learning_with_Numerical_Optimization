# Created by Ozan Bali
# 03.19.2025
## Newton - Raphson Algorithm For Roots

x = 3 # Initialize value for function
f = (x - 1) ** 2 * (x - 2) * (x - 3)  # Function
f1 = 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * (2 * x - 5)  # First derivative
f2 = 2 * (x - 2) * (x - 3) + 2 * (x - 1) * (2 * x - 5) + 2 * (x - 1) * (2 * x - 5) + 2 * (x - 1) ** 2 # Second derivate

dx = f1 / f2
iteration = 0 #Counter

print("--",iteration, " x: " ,x,"f: ",f,"f1: ",f1,"f2: ",f2,"dx: ",dx) # Initialize values

while abs(f1) > 1e-10:
    iteration += 1
    x = x + dx # Update value

    f = (x - 1) ** 2 * (x - 2) * (x - 3)
    f1 = 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * (2 * x - 5)
    f2 = 2 * (x - 2) * (x - 3) + 2 * (x - 1) * (2 * x - 5) + 2 * (x - 1) * (2 * x - 5) + 2 * (x - 1) ** 2

    dx = -f1 / f2 # Instability
    print("--", iteration, " x: ",round(x,4), "f: ", round(f,4), "f1: ", round(f1,4), "f2: ", round(f2,4), "dx: ", round(dx,4))