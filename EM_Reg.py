# ===============================================
# Import Packages and Functions
import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM
from   itertools            import combinations 
from   random               import sample


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
        
  