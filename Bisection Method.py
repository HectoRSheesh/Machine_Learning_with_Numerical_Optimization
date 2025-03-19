# Created by Ozan Bali
# 03.19.2025
# Bisection Method

x1 = -111111 # First root
x2 = 51111 # Second root
iteration = 0 # Counter

def calculate_derivate(x):
     return 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * (2 * x - 5) # First derivate for function

while True:
        if (calculate_derivate(x1) * calculate_derivate(x2)) < 0: # Answer should be in these given roots
            xk = x1 + (x2 - x1) / 2
            iteration += 1
            if calculate_derivate(xk) == 0 or abs(x2-x1) < 1e-9: # You can change parameter for abs
                print("--",iteration,'x_root:',round(xk,5)," f1:",round(calculate_derivate(xk),4))
                break
            elif calculate_derivate(xk) * calculate_derivate(x1) > 0:
                    x1 = xk
            else:
                    x2 = xk
                    print("--",iteration,'x_root:',round(xk,5)," f1:",round(calculate_derivate(xk),4))

        else:
            print('There is no root for these values...')
            break