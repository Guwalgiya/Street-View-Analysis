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
def EM_Reg(train_data, num_cluster):
    
    
    slope_bound = 100
    # ===============================================
    # Initialization
    slopes    = sample(range(-1 * slope_bound, slope_bound), num_cluster)
    slopes    = [slope / slope_bound for slope in slopes]
    intercept = sample(range(-1 * slope_bound, slope_bound), num_cluster)
    
    
    
    for data_point in train_data:
        a = 1
        
    best_labels = [randint(1, num_cluster) for data_point in train_data]
    
    a = 1
    if True:
      plt.figure()
      plt.axis([0, 540, 0, 540])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = best_labels, cmap= "viridis")
      plt.gca().invert_yaxis()
      
# ===============================================
# Debug
debug_data = np.load("dabug_data_1.npy")
EM_Reg(debug_data, 3)