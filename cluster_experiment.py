from random import randint
from sklearn.mixture import GMM
import matplotlib.pyplot as     plt
import numpy as np
# =============================================================================
# ks = [-0.5, -1, -2]
# bs = [150, 400, 1000]
# n = 6
# X1 = []
# X2 = []
# for i in range(len(ks)):
#     k  = ks[i]
#     b  = bs[i]
#     x1 = [0] * n
#     x2 = [0] * n
#     for j in range(n):
#         x1[j] = randint(0, 480)
#         x2[j] = int(k * x1[j] + b) + randint(-5, 5)
#     
#     X1 = X1 + x1
#     X2 = X2 + x2
# =============================================================================
lines = np.load("lines.npy")
X1 = []
X2 = []
for cur_line in lines:
    x1, y1, x2, y2 = cur_line[0]  
    if x1 <= 480 and x2 <= 480:
        X1.append(x1)
        X1.append(x2)
        X2.append(y1)
        X2.append(y2)



train_data = np.column_stack((X1, X2))

cluster = GMM(n_components = 3, covariance_type = "full")
labels  = cluster.fit_predict(train_data)
plt.figure()
plt.axis([0, 480 * 2, 0, 540])
plt.scatter(X1, X2, c = labels, cmap= "viridis")