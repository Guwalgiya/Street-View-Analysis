import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM

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
        scores            = {}
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
                scores[label] = score
                
                
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
        # Check if this is a good clustering
        avg_score = sum(scores.values()) / len(scores.keys())
        if abs(avg_score - 1) <= abs(best_avg_score - 1):
            
            
            # ===============================================
            # Update
            best_avg_score               = avg_score 
            best_cluster_dict            = {}
            best_cluster_dict["scores"]  = scores
            best_cluster_dict["X1_all"]  = X1_all
            best_cluster_dict["X2_all"]  = X2_all
            best_cluster_dict["labels"]  = labels
            best_cluster_dict["cluster"] = cluster
     
        
    # ===============================================
    # Try to reduce the number of clusters
    print(best_avg_score)
    print(best_cluster_dict["scores"])
    best_scores = best_cluster_dict["scores"]
    labels = best_cluster_dict["labels"]
    for vassal_label in list(set(labels)):
        for suzerain_label in list(set(labels)):
            print('------------------------------------')
            print(vassal_label, suzerain_label)
            temp_scores = best_scores.copy()
            if vassal_label != suzerain_label:
                print(temp_scores)

                vassal_indices   = np.where(labels == vassal_label)[0].tolist()
                suzerain_indices = np.where(labels == suzerain_label)[0].tolist()
                indices          = np.concatenate((vassal_indices, suzerain_indices))
                reg_data         = train_data[indices]
                reg              = LinearRegression().fit(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                k     = reg.coef_[0]
                b     = reg.intercept_
                score = reg.score(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                
                temp_scores[suzerain_label] = score
                print(temp_scores)
                temp_scores.pop(vassal_label, None)
                merged_avg_scores = sum(temp_scores.values()) / len(temp_scores.keys())
                print(temp_scores)
                print(merged_avg_scores)
                if abs(merged_avg_scores - 1) <= abs(best_avg_score - 1):
                    best_avg_score = merged_avg_scores
                    print(merged_avg_scores)
                
                
                
    # ===============================================
    # If we want to see clusters
    if if_show_cluster:
      plt.figure()
      plt.axis([0, width, 0, height])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = best_cluster_dict["labels"], cmap= "viridis")
      plt.gca().invert_yaxis()
        
      
    return best_cluster_dict["X1_all"], best_cluster_dict["X2_all"]
