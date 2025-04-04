import numpy as np
import math
import matplotlib.pyplot as plt

from ornekFonksiyon2 import f, hessian
from ornekFonksiyon2 import gradient as gradf


def GSmain(f, xk, pk):
    """
    Uses the golden-section search to find the minimum of the function f(xk + s*pk)
    over the interval [0,1].
    """
    # Define the lower and upper bounds of the search interval
    xbottom = 0
    xtop = 1
    dx = 1e-5

    # Compute the golden ratio values
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))  # Estimate number of iterations

    # Initialize the two points within the interval
    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f(xk + x1 * pk)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f(xk + x2 * pk)

    # Perform the golden-section search loop for N iterations
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

    return 0.5 * (x1 + x2)  # Return the midpoint of the final interval as the optimal step size


# --- STEP 1: Initialization ---

# Set the starting point
x = np.array([-5.4, 1.7])

# Define the maximum number of iterations
Nmax = 10000

# Termination criteria thresholds:
eps1 = 1e-10  # Threshold for change in function value
eps2 = 1e-10  # Threshold for change in the variable x
eps3 = 1e-10  # Threshold for the gradient norm (stationary point)

k = 0  # Iteration counter

# Set the initial metric matrix (M0) as the identity matrix
I = np.eye(2)
M = np.eye(2)

# Lists to record the history of iterations for plotting
x1_iter = [x[0]]
x2_iter = [x[1]]

# --- STEP 2: Iterative Process ---

while True:
    # Compute the gradient at the current point x_k
    grad_current = gradf(x)

    # Check termination if the gradient norm is sufficiently small
    if np.linalg.norm(grad_current) < eps3:
        print("Convergence reached: Gradient norm < eps3")
        break

    # Terminate if maximum iterations have been exceeded
    if k >= Nmax:
        print("Maximum number of iterations exceeded")
        break

    # Check if the metric matrix M is positive definite by inspecting its eigenvalues
    eig_vals = np.linalg.eigvals(M)
    if np.min(eig_vals) > eps1:
        # If M is positive definite, choose the search direction p_k = -M * grad_current
        pk = -np.dot(M, grad_current)
    else:
        # Otherwise, add a small multiple of the identity matrix to ensure positive definiteness
        mu = abs(np.min(eig_vals)) + 0.001
        pk = -np.dot(M + mu * I, grad_current)

    # Determine the step size s_k using a one-dimensional golden-section search along direction p_k
    sk = GSmain(f, x, pk)

    # Store the current x and function value for later comparison
    x_old = x.copy()
    f_old = f(x)

    # Update the variable: x_(k+1) = x_k + s_k * p_k
    x = x + sk * pk

    # Append the new point to the iteration history for plotting
    x1_iter.append(x[0])
    x2_iter.append(x[1])

    # Compute the new gradient at the updated point x_(k+1)
    grad_new = gradf(x)

    # Terminate if the change in the function value is below the threshold
    if abs(f(x) - f_old) < eps1:
        print("Function value change is below threshold: |f(x_k+1) - f(x_k)| < eps1")
        break
    # Terminate if the change in the variable x is below the threshold
    if np.linalg.norm(x - x_old) < eps2:
        print("Change in variables is below threshold: ||x_k+1 - x_k|| < eps2")
        break

    # --- Update the Metric Matrix using the DFP formula ---
    # Dx is the change in x (i.e., s_k * p_k)
    Dx = (sk * pk).reshape(-1, 1)
    # y is the difference in gradients: grad f(x_(k+1)) - grad f(x_k)
    y = (grad_new - grad_current).reshape(-1, 1)

    # Compute denominators for the update formulas; avoid division by near-zero values
    denom_A = np.dot(Dx.T, y).item()
    denom_B = np.dot(y.T, np.dot(M, y)).item()

    if abs(denom_A) > 1e-12 and abs(denom_B) > 1e-12:
        A = np.dot(Dx, Dx.T) / denom_A
        B = np.dot(np.dot(M, y), (np.dot(M, y)).T) / denom_B
        # Update the metric matrix M using the DFP update formula
        M = M + A - B
    else:
        # If denominators are too small, skip the metric update
        pass

    k += 1  # Increment iteration counter
    print("Iteration:", k, " Step size:", sk, " x:", np.round(x, 4), " f(x):", np.round(f(x), 4))

# --- STEP 3: Plot the Iteration History ---
plt.figure(figsize=(8, 6))
plt.plot(x1_iter, x2_iter, marker='o', linestyle='-', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Iteration History')
plt.grid(True)
plt.show()

"""
import numpy as np
import math
import matplotlib.pyplot as plt

from ornekFonksiyon2 import f, hessian
from ornekFonksiyon2 import gradient as gradf


def GSmain(f, xk, pk):

    xbottom = 0
    xtop = 1
    dx = 1e-5
    # Altın oran hesaplaması:
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))

    # İlk iki noktanın belirlenmesi:
    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f(xk + x1 * pk)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f(xk + x2 * pk)

    # Golden-section döngüsü:
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

    return 0.5 * (x1 + x2)


# --- ADIM 1: Başlangıç ---

# Başlangıç noktası
x = np.array([-5.4, 1.7])

# Maksimum iterasyon sayısı
Nmax = 10000

# Sonlandırma kriterleri
eps1 = 1e-10  # Fonksiyon değişim eşiği
eps2 = 1e-10  # x değişim eşiği (değişken değişmiyor durumu)
eps3 = 1e-10  # Gradyan normu eşiği (durağan nokta)

k = 0  # Iterasyon sayacı

# Başlangıç metriği (M0) – burada birim matris seçildi
I = np.eye(2)
M = np.eye(2)

# İterasyon geçmişini tutmak (grafik için)
x1_iter = [x[0]]
x2_iter = [x[1]]

# --- ADIM 2: İterasyon Döngüsü ---

while True:
    # xk noktasındaki gradyantı hesapla
    grad_current = gradf(x)

    # Sonlandırma: Gradyan normu yeterince küçükse
    if np.linalg.norm(grad_current) < eps3:
        print("Durağan noktaya gelindi: Grad f(x) normu < eps3")
        break

    # Maksimum iterasyon kontrolü
    if k >= Nmax:
        print("Maksimum iterasyon sayısı aşıldı")
        break

    # Matris M'nin pozitif tanımlı olup olmadığını kontrol et
    eig_vals = np.linalg.eigvals(M)
    if np.min(eig_vals) > eps1:
        pk = -np.dot(M, grad_current)
    else:
        # Uygun bir matris ilavesiyle (mu*I) pozitif tanımlılık sağla
        mu = abs(np.min(eig_vals)) + 0.001
        pk = -np.dot(M + mu * I, grad_current)

    # Bir boyutlu optimizasyonla (Golden-Section) sₖ (adım uzunluğu) belirle
    sk = GSmain(f, x, pk)

    # Güncelleme öncesi değerleri sakla
    x_old = x.copy()
    f_old = f(x)

    # xₖ₊₁ = xₖ + sₖ * pₖ
    x = x + sk * pk

    # İterasyon geçmişine ekle (grafik için)
    x1_iter.append(x[0])
    x2_iter.append(x[1])

    # Yeni gradyantı hesapla
    grad_new = gradf(x)

    # Sonlandırma: Fonksiyon değeri ve x değişiklikleri
    if abs(f(x) - f_old) < eps1:
        print("Fonksiyon değeri değişmiyor: |f(xₖ₊₁) - f(xₖ)| < eps1")
        break
    if np.linalg.norm(x - x_old) < eps2:
        print("Değişkenlerde değişiklik yok: ||xₖ₊₁ - xₖ|| < eps2")
        break

    # Metrik güncellemesi için:
    # updatedx = sₖ * pₖ (adım değişimi)
    Dx = (sk * pk).reshape(-1, 1)
    # y = grad f(xₖ₊₁) - grad f(xₖ)
    y = (grad_new - grad_current).reshape(-1, 1)

    # Bölge kontrolü (bölme hatasından kaçınmak için)
    denom_A = np.dot(Dx.T, y).item()
    denom_B = np.dot(y.T, np.dot(M, y)).item()

    if abs(denom_A) > 1e-12 and abs(denom_B) > 1e-12:
        A = np.dot(Dx, Dx.T) / denom_A
        B = np.dot(np.dot(M, y), (np.dot(M, y)).T) / denom_B
        M = M + A - B
    else:
        # Eğer bölge çok küçükse M'yi güncelleme yapmadan bırakabiliriz.
        pass

    k += 1
    print("k:", k, " sₖ:", sk, " x:", np.round(x, 4), " f(x):", np.round(f(x), 4))

# --- ADIM 3: Grafiğe Dökme ---
plt.figure(figsize=(8, 6))
plt.plot(x1_iter, x2_iter, marker='o', linestyle='-', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('İterasyonlar')
plt.grid(True)
plt.show()
"""