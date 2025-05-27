import sympy as sp

# 1. Değişkenleri tanımla
x, y , z= sp.symbols('x y z')

# 2. Fonksiyon tanımı
f = (x - 2* y + z + 1)**2 + (x + y - z + 3)**2 + (-2 * x + y - z)**2 # Buraya istediğin fonksiyonu yazabilirsin

# 3. Gradyan vektörü
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
df_dz = sp.diff(f, z)

grad_f = sp.Matrix([df_dx, df_dy , df_dz])

# 4. Gradyan = 0 ⇒ Kritik noktalar
kritik_noktalar = sp.solve([df_dx, df_dy], (x, y), dict=True)

# 5. Hessian oluştur
f_xx = sp.diff(f, x, x)
f_xy = sp.diff(f, x, y)
f_yx = sp.diff(f, y, x)
f_yy = sp.diff(f, y, y)
f_xz = sp.diff(f, x, z)
f_yz = sp.diff(f, y, z)
H = sp.Matrix([[f_xx, f_xy , f_xz],
               [f_yx, f_yy , f_yz]])

# 6. Yazdırmalar
print("Fonksiyon f(x, y):")
sp.pprint(f)

print("\nGradyan vektörü ∇f(x, y):")
sp.pprint(grad_f)

print("\nGradyanı sıfır yapan kritik noktalar:")
for nokta in kritik_noktalar:
    sp.pprint(nokta)

# 7. Kritik noktalarda Hessian ve özdeğer analizi
for i, nokta in enumerate(kritik_noktalar, 1):
    print(f"\n👉 Kritik Nokta {i}: {nokta}")
    H_at_point = H.subs(nokta)
    print("Hessian @ Nokta:")
    sp.pprint(H_at_point)

    #eigenvals = list(H_at_point.eigenvals().keys())

   # print("Özdeğerler:", eigenvals)

    # 8. Nokta türünü belirleme
    #if all(ev.is_positive for ev in eigenvals):
        #print("⏹️ Bu nokta: LOKAL MİNİMUM")
    #elif all(ev.is_negative for ev in eigenvals):
        #print("⏹️ Bu nokta: LOKAL MAKSİMUM")
    #elif any(ev.is_positive for ev in eigenvals) and any(ev.is_negative for ev in eigenvals):
        #print("⏹️ Bu nokta: SADDLE POINT (Eyer Noktası)")
    #else:
        #print("⚠️ Bu noktada karar verilemez (belirsiz özdeğer)")
