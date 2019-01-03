# ===============================================
# Import Packages and Functions
from   sklearn.linear_model import LinearRegression
from   sklearn.mixture      import GaussianMixture
from   itertools            import combinations 
import matplotlib.pyplot    as     plt
import numpy                as     np
import cv2


# ===============================================
# Function Draw
def lineDetector(original_image, input_image, draw_parameters_bundle, job_type):
    
    
    # ===============================================
    # Get Parameters
    if_show_original_image = draw_parameters_bundle["if_show_original_image"]
    if_show_R_cluster      = draw_parameters_bundle["if_show_R_cluster"]
    if_show_L_cluster      = draw_parameters_bundle["if_show_L_cluster"]
    if_show_scatters       = draw_parameters_bundle["if_show_scatters"]
    min_line_length        = draw_parameters_bundle["min_len"]  
    slope_threshold        = draw_parameters_bundle["s_threshold"]
    painting_color         = draw_parameters_bundle["color"] 
    drawing_height         = draw_parameters_bundle["draw_height"]
    line_channel           = draw_parameters_bundle["channel"]
    max_line_gap           = draw_parameters_bundle["max_gap"]  
    threshold              = draw_parameters_bundle["h_threshold"] 
    data_type              = draw_parameters_bundle["d_type"]  
    thickness              = draw_parameters_bundle["thick"] 
    theta                  = draw_parameters_bundle["theta"]     
    rho                    = draw_parameters_bundle["rho"]    
    
    
    # ===============================================
    if if_show_original_image:
        plt.figure()
        plt.imshow(original_image)
    
    
    # ===============================================
    # Get Lines of the Pictures
    lines = cv2.HoughLinesP(input_image,  rho, theta, 
                            threshold,    np.array([]), 
                            minLineLength = min_line_length, 
                            maxLineGap    = max_line_gap)
           
    
    # ===============================================
    # Draw those scatters
    if if_show_scatters:
        temp_image = original_image.copy()
        plt.figure()
        plt.imshow(temp_image)
        plt.scatter(np.concatenate((lines[:, 0, 0], lines[:, 0, 2])), 
                    np.concatenate((lines[:, 0, 1], lines[:, 0, 3])))
        
        
    # ===============================================
    # Initializing
    R_X      = []
    R_Y      = []
    L_X      = []
    L_Y      = []


    # ===============================================
    # Frame Information 
    width    = original_image.shape[1]
    height   = original_image.shape[0]
    center_x = width  / 2
    
    
    # ===============================================
    # Get all scatters from lines
    for cur_line in lines:
        x1, y1, x2, y2 = cur_line[0]  
                
        
        # ===============================================
        # Calculate Slopes
        slope = (y2 - y1) / (x2 - x1)    
        
        
        # ===============================================
        # Seperate Points
        if abs(slope) > slope_threshold:
                      
            
            # ===============================================
            # Seperate Points: Points to the L
            if slope < 0 and x1 < center_x and x2 < center_x:
                L_X.append(x1)
                L_X.append(x2)
                L_Y.append(y1)
                L_Y.append(y2)
                
                
            # ===============================================
            # Seperate Points: Points to the R
            elif slope > 0 and x1 > center_x and x2 > center_x:
                R_X.append(x1)
                R_X.append(x2)
                R_Y.append(y1)
                R_Y.append(y2)
                
        else:
            
            # ===============================================
            # Seperate Points: Points to the L
            if slope < 0 and x1 < center_x and x2 < center_x:
                L_X.append((x1 + x2) / 2)
                L_Y.append((y1 + y2) / 2)
                
                
            # ===============================================
            # Seperate Points: Points to the R
            elif slope > 0 and x1 > center_x and x2 > center_x:
                R_X.append((x1 + x2) / 2)
                R_Y.append((y1 + y2) / 2)


    # ===============================================
    # Prepare for drawing
    Y1          = input_image.shape[0]
    Y2          = input_image.shape[0] * drawing_height
    lines_image = np.zeros((*input_image.shape, line_channel), dtype = data_type)
    
    
    # ===============================================
    # Make 2-D points
    train_data_L = np.column_stack((L_X, L_Y))
    train_data_R = np.column_stack((R_X, R_Y))
    
    
    if job_type == "Video":
        # ===============================================
        dict_train_data_L = np.load("train_data_L.npy").item()
        dict_train_data_R = np.load("train_data_R.npy").item()
        
        
        # ===============================================
        initial_data_array    = np.array([])
        initial_data_array    = initial_data_array.reshape((len(initial_data_array), 2))
        previous_train_data_L = initial_data_array
        previous_train_data_R = initial_data_array
        
        
        # ===============================================
        # Get more datapoints from previous frame when processing videos
        for key in dict_train_data_L.keys():
            previous_train_data_L = np.concatenate((previous_train_data_L, dict_train_data_L[int(key)]))
            previous_train_data_R = np.concatenate((previous_train_data_R, dict_train_data_R[int(key)]))
        
        
        # ===============================================        
        max_key = max(dict_train_data_L.keys())
        min_key = min(dict_train_data_L.keys())
        
        
        # ===============================================
        dict_train_data_L.pop(min_key)
        dict_train_data_R.pop(min_key)
        
        
        # ===============================================
        dict_train_data_L[max_key + 1] = train_data_L
        dict_train_data_R[max_key + 1] = train_data_R
        
        
        # ===============================================
        np.save("train_data_L.npy", dict_train_data_L)
        np.save("train_data_R.npy", dict_train_data_R)
    
    
        # ===============================================
        # Get more datapoints from previous frame when processing videos
        train_data_L = np.concatenate((train_data_L, previous_train_data_L))
        train_data_R = np.concatenate((train_data_R, previous_train_data_R))


    # ===============================================
    # Get Crucial Points for the both sections
    R_X1_all, R_X2_all, num_R_lines = clusteringPoints(train_data_R, if_show_R_cluster, Y1, Y2, height, width, "R")
    L_X1_all, L_X2_all, num_L_lines = clusteringPoints(train_data_L, if_show_L_cluster, Y1, Y2, height, width, "L")
    
    
    # ===============================================
    # Concatenate Vectors
    X1_all = np.concatenate((L_X1_all, R_X1_all))
    X2_all = np.concatenate((L_X2_all, R_X2_all))
    
    
    # ===============================================
    # Draw Lines
    for X1, X2 in np.column_stack((X1_all, X2_all)):
        lines_image = cv2.line(lines_image, (int(X1),  int(Y1)), (int(X2),  int(Y2)), painting_color, thickness)
    
    
    # ===============================================
    # Add number of lines to a frame
    cv2.putText(lines_image, str(num_L_lines + num_R_lines), (150, 150), cv2.LINE_AA, 5, painting_color, thickness)
    
    
    # ===============================================
    return lines_image



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
        cluster = GaussianMixture(n_components = num_cluster, covariance_type = "full")
        gmm     = cluster.fit(train_data)
        labels  = gmm.predict(train_data)
        
        
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