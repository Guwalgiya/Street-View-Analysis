# ===============================================
# Import Packages and Functions
import matplotlib.pyplot    as     plt
import numpy                as     np
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GMM
from   itertools            import combinations 


# ===============================================
# Use GMM to classify lines
def clusteringPoints(train_data, if_show_cluster, Y1, Y2, height, width, side):
    
    
    # ===============================================
    num_cluster_choices = [1, 2, 3, 4] 
        
        
    # ===============================================
    # Initialize Average Score  
    best_avg_score      = np.inf
    
    
    # ===============================================
    # Start to select number of clusters
    for num_cluster in num_cluster_choices:

        
        # ===============================================
        # Do Clustering
        cluster = GMM(n_components = num_cluster, covariance_type = "full")
        labels  = cluster.fit_predict(train_data)
        
        
        # ===============================================
        # Prepare Validation
        scores            = []
        X1_all            = []
        X2_all            = []
        
        
        # ===============================================
        # Start Validation
        for label in range(num_cluster):
            indices = np.where(labels == label)[0].tolist()


            # ===============================================
            # Regression on clusters that have more than 1 point
            if len(indices) > 1:
                reg_data = train_data[indices]
                reg      = LinearRegression().fit(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
                
                
                # ===============================================
                # Get coefficient
                k = reg.coef_[0]
                b = reg.intercept_
                
                
                # ===============================================
                # Avoid Bad k-values
                if (side == "L" and k > 0) or (side == "R" and k < 0):
                    scores.append([-1, 1000])
                    break
                
                
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
        
        
        # ===============================================
        # Check if this is a good clustering
        avg_score = sum(pair[1] for pair in scores) / len(scores)

        if abs(avg_score - 1) <= abs(best_avg_score - 1):
            
            
            # ===============================================
            # Update
            best_avg_score               = avg_score 
            best_scores                  = scores
            best_num_cluster            = num_cluster
            best_X1_all  = X1_all
            best_X2_all  = X2_all
            best_labels  = labels
     
  
    # ===============================================
    # Try to reduce the number of clusters
    while (len(best_scores) > 1):
        
        
        # ===============================================
        # Initialization & n choose 2
        if_reduced = False
        label_comb = combinations(list(set(best_labels)), 2)
        
        
        # ===============================================
        # True to merge vassal_label to suzerain_label
        for vassal_label, suzerain_label in label_comb:
            
            
            # ===============================================
            # Get Merged Indices
            vassal_indices   = np.where(best_labels == vassal_label)[0].tolist()
            suzerain_indices = np.where(best_labels == suzerain_label)[0].tolist()
            indices          = np.concatenate((vassal_indices, suzerain_indices))
            
            
            # ===============================================
            # Do Regression for merged Data
            reg_data = train_data[indices]
            reg      = LinearRegression().fit(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
            score    = reg.score(reg_data[:, 0].reshape(-1, 1), reg_data[:, 1])
            
            
            # ===============================================
            # After merging two clusters, what are the scores for all clusters?
            scores    = [pair for pair in best_scores 
                         if (pair[0] != vassal_label and pair[0] != suzerain_label)]
            scores.append([suzerain_label, score])
            
        

            # ===============================================
            # Get Average
            avg_score = sum(pair[1] for pair in scores) / len(scores)
            
            

            # ===============================================
            # Also need to worry about k
            temp_k = reg.coef_[0]
            if (side == "L" and temp_k <=0) or (side == "R" and temp_k >= 0):
                valid_k = True
            else:
                valid_k = False
                
    
            # ===============================================
            # Does this merging actuall help us??? 
            # 0.03 is tolerance because we want to merge some clusters
            if (abs(avg_score - 1) <= abs(best_avg_score - 1) + 0.03) and valid_k:
                
                
                # ===============================================
                # Save information if  we want to use this merge later
                k              = temp_k
                b              = reg.intercept_
                new_X1         = (Y1 - b) / k
                new_X2         = (Y2 - b) / k
                removed_label  = vassal_label
                merged_label   = suzerain_label
                
                
                # ===============================================
                # Useful for next loop round
                best_avg_score = avg_score
                if_reduced     = True
         
            
        # ===============================================
        # If num of clusters is reduced, we use the best merging
        if if_reduced:        
            
            
            # ===============================================
            # Update information for drawing
            best_num_cluster  = best_num_cluster  - 1
            best_labels = [merged_label if item == removed_label else item for item in best_labels]
            best_X1_all = [pair for pair in best_X1_all if (pair[0] != removed_label and pair[0] != merged_label)]
            best_X2_all = [pair for pair in best_X2_all if (pair[0] != removed_label and pair[0] != merged_label)]
            best_X1_all.append([merged_label, new_X1])
            best_X2_all.append([merged_label, new_X2])
        
        
        # ===============================================
        # If not reduced, we just what we have before
        else:
            break


    # ===============================================
    # If we want to see clusters
    if if_show_cluster:
      plt.figure()
      plt.axis([0, width, 0, height])
      plt.scatter(train_data[:, 0], train_data[:, 1], c = best_labels, cmap= "viridis")
      plt.gca().invert_yaxis()
        
      
    return [pair[1] for pair in best_X1_all], [pair[1] for pair in best_X2_all], len(list(set(best_labels)))
