"""
import numpy as np
import math
import matplotlib.pyplot as plt

from ornekFonksiyon2 import f, hessian, jacobian, error
from ornekFonksiyon2 import gradient as gradf


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

    return result # Return the midpoint of the final interval as the optimal step size

x = np.array([-5.4, 1.7])
x1 = [x[0]]
x2 = [x[1]]
muk = 1
mumin = 1e-20
mumax = 1e+10
muscale = 10
Nmax = 10000

eps1 = 1e-10  # Threshold for change in function value
eps2 = 1e-10  # Threshold for change in the variable x
eps3 = 1e-10  # Threshold for the gradient norm (stationary point)

k = 0  # Iteration counter

# Set the initial metric matrix (M0) as the identity matrix
I = np.identity(2)
updatedx = np.array([1e10, 1e10])

C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x ) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

while not (C1 or C2 or C3 or C4):
    k += 1
    J = jacobian(x)
    e = error(x)
    A = np.dot(np.matrix.transpose(J), J)
    B = np.dot(np.matrix.transpose(J),e.reshape(-1,1))
    loop = True
    while loop:
        zk = -np.dot(np.linalg.inv(A + muk * I),B)
        zk = np.array(zk).reshape(-1,)
        if f(x + zk) < f(x):
            sk = 1.0
            x = x + sk*zk
            x = np.array(x)
            muk = muk / muscale
            loop = False
        else:
            muk = muk / muscale
            if mumax < muk:
                loop = False
    print("k:",k," sk:",round(sk,4)," x1:",round(x[0],4)," x2:",round(x[1],4)," f(x)",round(f(x),4))
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    updatedx = 1*x
    x1.append(x[0])
    x2.append(x[1])

if C1:
    print("Maksimum iterasyonu ulaşıldı")
if C2:
    print("Fonksiyon değişmiyor")
if C3:
    print("Değişkenler değişmiyor")
if C4:
    print("Durağan noktaya gelindi")

plt.figure(figsize=(8, 6))
plt.plot(x1, x2, marker='o', linestyle='-', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Iteration History')
plt.grid(True)
plt.show()
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# ornekFonksiyon2 modülündeki fonksiyonları içe aktarıyoruz.
from ornekFonksiyon2 import f, jacobian, error, gradient as gradf


def GSmain(f, xk, pk):
    """
    Golden Section Search yöntemiyle [0,1] aralığında
    optimal adım boyunu bulur.
    """
    # xk ve pk'nın 1D array olduğundan emin olun.
    xk = np.asarray(xk).flatten()
    pk = np.asarray(pk).flatten()

    xbottom, xtop = 0, 1
    dx = 1e-5
    alpha = (1 + math.sqrt(5)) / 2  # Altın oran
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))

    # İlk iki nokta ve fonksiyon değerleri
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

    return 0.5 * (x1 + x2)


# --- Başlangıç Ayarları ---
x = np.array([0,0])  # Başlangıç noktası (1D array, uzunluk=2)
u = 1.0  # Başlangıç damping (u) değeri
umin, umax = 1e-20, 1e+10  # u için sınır değerleri
uscal = 10.0  # u'nun güncelleme ölçeği
Nmax = 10000  # Maksimum iterasyon sayısı

# Sonlandırma kriterleri
e1, e2, e3 = 1e-10, 1e-10, 1e-10

k = 0  # İterasyon sayacı
I = np.identity(len(x))  # Boyut uygun kimlik matrisi

while True:
    k += 1
    if k > Nmax:
        print("Maksimum iterasyona ulaşıldı.")
        break

    # x'nin mevcut fonksiyon değeri
    f_x = f(x)

    # Jacobian ve hata vektörünü alıp, açıkça array'e çeviriyoruz.
    J = np.asarray(jacobian(x))
    e_vec = np.asarray(error(x)).flatten()  # 1D hale getiriyoruz

    A = J.T @ J
    B = J.T @ e_vec.reshape(-1, 1)

    updated = False  # İç döngüde güncelleme yapılıp yapılmadığını kontrol edeceğiz.
    while True:
        try:
            inv_term = np.linalg.inv(A + u * I)
        except np.linalg.LinAlgError as err:
            print("Matrisi ters çevirme hatası:", err)
            break

        zk = -inv_term @ B
        zk = np.asarray(zk).flatten()  # 1D güncelleme vektörü

        # Yeni aday x; hem x hem de zk'nın 1D olduğundan emin oluyoruz.
        new_x_candidate = x.flatten() + zk

        if f(new_x_candidate) < f_x:
            # İyileşme varsa Golden Section Search ile optimal adım (sk) bulunuyor.
            pk = zk.copy()
            sk = GSmain(f, x, pk)
            x_new = x.flatten() + sk * pk
            # Başarılı adım sonrası damping faktöre düşürme uyguluyoruz.
            u = max(umin, u / uscal)
            updated = True
            break
        else:
            # Adım işe yaramıyorsa damping faktörü artırılıyor.
            u *= uscal
            if not (umin < u < umax):
                break

    if not updated:
        print("İç döngü güncellemesi yapılamadı, damping faktörü sınırları aşıldı.")
        break

    # Yeni x için fonksiyon değeri
    f_new = f(x_new)
    # Konverjans kontrolü
    C2 = abs(f_new - f_x) < e1
    C3 = np.linalg.norm(x_new - x.flatten()) < e2
    C4 = np.linalg.norm(gradf(x_new)) < e3
    C5 = (umin < u < umax)  # damping faktörünün sınırlar içinde olup olmadığı

    print(f"k:{k}, sk:{round(sk, 6)}, x1:{round(x_new[0], 6)}, x2:{round(x_new[1], 6)}, f(x):{round(f_new, 6)}")

    if C2:
        print("Fonksiyon değişmiyor")
        break
    if C3:
        print("Değişkenler değişmiyor")
        break
    if C4:
        print("Durağan noktaya gelindi")
        break
    if not C5:
        print("Damping faktörü sınırları dışına çıktı, algoritma durduruluyor.")
        break

    # x'i güncelle
    x = x_new.copy()

# İterasyon ilerleyişini görselleştirmek için (tek son adım gösteriliyor)
plt.figure(figsize=(8, 6))
plt.scatter(x_new[0], x_new[1], c='red', label="Son x")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Son Nokta')
plt.grid(True)
plt.legend()
plt.show()

