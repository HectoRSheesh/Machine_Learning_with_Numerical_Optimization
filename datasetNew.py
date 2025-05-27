import numpy as np
import math
import matplotlib.pyplot as plt

# Verinin oluşturulması
ti = np.arange(-4, 4, 0.4)
ti = ti.reshape(-1, 1)  # 2D hale getiriyoruz
yi = [3.6294 * math.exp(-0.116 * t[0]) + np.random.random() * 0.10 for t in ti]

# Grafik çizimi
plt.scatter(ti, yi, color='darkred')
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Dataset 4', fontstyle='italic')
plt.grid(color='green', linestyle='--', linewidth=0.1)
plt.show()
