# ===============================================
# Import Packages and Functions
import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM
from   itertools            import combinations 


# ===============================================
# Use GMM to classify lines
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
        scores            = []
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
                score = reg.score(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                
                
                # ===============================================
                # Scores          
                scores.append([label, score])
                
                
                # ===============================================
                # Get drawing points for X coordinate
                X1 = (Y1 - b) / k
                X2 = (Y2 - b) / k
                
                
                # ===============================================
                # Append X coordinates to lists
                X1_all.append([label, X1])
                X2_all.append([label, X2])
                
                
                # ===============================================
                # Update Info
                num_valid_cluster = num_valid_cluster + 1
        
        
        # ===============================================
        # Check if this is a good clustering
        avg_score = sum(pair[1] for pair in scores) / len(scores)
        if abs(avg_score - 1) <= abs(best_avg_score - 1):
            
            
            # ===============================================
            # Update
            best_avg_score               = avg_score 
            best_scores                  = scores
            best_X1_all  = X1_all
            best_X2_all  = X2_all
            best_labels  = labels
     
        
    # ===============================================
    # Try to reduce the number of clusters
    can_be_reduced = True
    while (len(best_scores) > 1):
        if_reduced = False
        label_comb = combinations(list(set(best_labels)), 2)
        for vassal_label, suzerain_label in label_comb:
            vassal_indices   = np.where(best_labels == vassal_label)[0].tolist()
            suzerain_indices = np.where(best_labels == suzerain_label)[0].tolist()
            indices          = np.concatenate((vassal_indices, suzerain_indices))
    
            reg_data         = train_data[indices]
            reg              = LinearRegression().fit(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
            k     = reg.coef_[0]
            b     = reg.intercept_
            score = reg.score(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
            new_scores = [pair for pair in best_scores if (pair[0] != vassal_label and pair[0] != suzerain_label)]
            new_scores.append([suzerain_label, score])
            avg_score = sum(pair[1] for pair in new_scores) / len(new_scores)
            if abs(avg_score - 1) <= abs(best_avg_score - 1) + 0.05:
                new_X1 = (Y1 - b) / k
                new_X2 = (Y2 - b) / k
                best_avg_score = avg_score
                
                removed_label  = vassal_label
                merged_label   = suzerain_label
                if_reduced        = True
        
        
        if if_reduced:        
            best_scores = new_scores
            best_labels = [merged_label if item == removed_label else item for item in best_labels]
            best_X1_all = [pair for pair in best_X1_all if (pair[0] != removed_label and pair[0] != merged_label)]
            best_X1_all.append([merged_label, new_X1])
            best_X2_all = [pair for pair in best_X2_all if (pair[0] != removed_label and pair[0] != merged_label)]
            best_X2_all.append([merged_label, new_X2])
        else:
            break
        
    # ===============================================
    # If we want to see clusters
    if if_show_cluster:
      plt.figure()
      plt.axis([0, width, 0, height])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = best_labels, cmap= "viridis")
      plt.gca().invert_yaxis()
        
      
    return [pair[1] for pair in best_X1_all], [pair[1] for pair in best_X2_all], len(best_scores)
