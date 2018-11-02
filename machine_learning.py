import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM
from   math                 import floor

def clusteringPoints(train_data, if_show_cluster, Y1, Y2, height, width):
    num_cluster_choices = [1, 2, 3, 4] 
    best_cluster_dict   = {}
    best_avg_score      = np.inf
    
    # ===============================================
    # Start to select number of clusters
    for num_cluster in num_cluster_choices:

        
        # ===============================================
        # Do Clustering
        cluster     = GMM(n_components = num_cluster, covariance_type = "full")
        labels      = cluster.fit_predict(train_data)
        
        
        # ===============================================
        # Prepare Validation
        num_valid_cluster = 0
        total_score       = 0
        X1_all            = []
        X2_all            = []
        
        
        # ===============================================
        # Start Validation
        for label in range(num_cluster):
            indices  = np.where(labels == label)[0].tolist()


            # ===============================================
            # Regression on clusters that have more than 1 point
            if len(indices) > 1:
                reg_data = train_data[indices]
                reg      = LinearRegression().fit(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                
                
                # ===============================================
                # Get coefficient
                k     = reg.coef_[0]
                b     = reg.intercept_
                
                
                # ===============================================
                # Get coefficient
                score       = reg.score(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                total_score = total_score + score
                
                
                # ===============================================
                # Get drawing points for X coordinate
                X1 = (Y1 - b) / k
                X2 = (Y2 - b) / k
                
                # ===============================================
                # Append X coordinates to lists
                X1_all.append(X1)
                X2_all.append(X2)
                
                
                # ===============================================
                # Update Info
                num_valid_cluster = num_valid_cluster + 1
        
        
        # ===============================================
        # Check if this is a good cluster
        avg_score = total_score / num_valid_cluster
        if abs(avg_score - 1) <= abs(best_avg_score - 1):
            
            
            # ===============================================
            # Update
            best_avg_score               = avg_score 
            best_cluster_dict            = {}
            best_cluster_dict["X1_all"]  = X1_all
            best_cluster_dict["X2_all"]  = X2_all
            best_cluster_dict["labels"]  = labels
            best_cluster_dict["cluster"] = cluster
            

    # ===============================================
    # If we want to see clusters
    if if_show_cluster:
      plt.figure()
      plt.axis([0, width, 0, height])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = best_cluster_dict["labels"], cmap= "viridis")
      plt.gca().invert_yaxis()
        
      
    return best_cluster_dict["X1_all"], best_cluster_dict["X2_all"]
# =============================================================================
#     d = {}
#    
#  
#     for i in range(num_clus):
#         indices  = np.where(labels == i)[0].tolist()
#         print(indices)
#         if len(indices) > 1:
#             reg_data = train_data[indices]
#             print(reg_data)
#             k, b     = np.polyfit(reg_data[:, 0], reg_data[:, 1], 1)
#             d[k]     = reg_data
#             draw_X1 = (draw_Y1 - b) / k
#             draw_X2 = (draw_Y2 - b) / k
#             line_image = line(line_image, (int(draw_X1),  int(draw_Y1)), (int(draw_X2),  int(draw_Y2)), painting_color, thickness)
#  
# =============================================================================
