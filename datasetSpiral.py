import numpy as np
import matplotlib.pyplot as plt

B = 1.8  # sayı arttıkça karmaşıklık artar spiraller yaklaşır
N = 100  # veri sayısı

Tall = np.array([]).reshape(2, -1)
for i in range(0, int(N / 2)):
    theta = np.pi / 2 + (i - 1) * ((2 * B - 1) / N) * np.pi
    A = np.array([theta * np.cos(theta), theta * np.sin(theta)]).reshape(2, 1)
    Tall = np.hstack((Tall, A))

Tall = np.hstack((Tall, -Tall))
Tmax = np.pi / 2 + ((N / 2 - 1) * (2 * B - 1) / N) * np.pi
ti = Tall.transpose() / Tmax
yi = np.hstack((np.ones(int(N / 2)), -np.ones(int(N / 2))))

plt.scatter(ti[:int(N / 2), 0], ti[:int(N / 2), 1], color='g', s=1, marker='o', alpha=0.99)
plt.scatter(ti[int(N / 2):, 0], ti[int(N / 2):, 1], color='r', s=1, marker='x', alpha=0.99)
plt.xlabel('$t_1$')
plt.ylabel('$t_2$')
plt.title('Dataset Spiral', fontstyle='italic')
plt.grid(color='green', linestyle='--', linewidth=0.1)
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
plt.show()
