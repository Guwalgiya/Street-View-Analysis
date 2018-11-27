# ===============================================
# Import Packages and Functions
import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM
from   itertools            import combinations 
from   random               import sample, randint


# ===============================================
# Use GMM to classify lines
# def EM_Reg(train_data, num_cluster):
    
train_data = np.load("dabug_data_1.npy")
num_cluster = 3 
slope_bound = 100
sigma       = 0.5
# ===============================================
# Initialization
slopes    = sample(range(-1 * slope_bound, slope_bound), num_cluster)
slopes    = [slope / slope_bound for slope in slopes]
intercept = sample(range(-1 * slope_bound, slope_bound), num_cluster)
#intercept = [0] * num_cluster
#labels = [randint(1, num_cluster) for data_point in train_data]
labels = [0] * len(train_data)
itr = 1
while itr < 2:
    for i in range(len(train_data)):
        x        = train_data[i][0] 
        y        = train_data[i][1] 
        residual = np.multiply(slopes, x) + intercept - y
        #exp_residual = np.exp(-1 * np.power(residual, 2) / (2 * np.power(sigma, 2)))
        #weights  = np.divide(exp_residual, sum(exp_residual))
        #labels[i] = np.argmax(weights)
        labels[i]  = np.argmin(residual)
    
    for label in range(num_cluster):
        indices = labels.index(label)
        reg_data = train_data[indices]
        a = 1
    print(labels)
    if True:
      plt.figure()
      plt.axis([0, 540, 0, 540])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = labels, cmap= "viridis")
      plt.gca().invert_yaxis()
      
# ===============================================
# Debug
# debug_data = np.load("dabug_data_1.npy")
# EM_Reg(debug_data, 3)