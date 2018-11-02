# ===============================================
# Import Packages and Functions
from   cv2               import HoughLinesP, line, addWeighted
from   machine_learning  import clusteringPoints
import numpy             as     np
import matplotlib.pyplot as     plt


# ===============================================
# Function Draw
def draw(original_image, input_image, draw_parameters_bundle):
    
    # ===============================================
    # Get Parameters
    if_show_right_cluster = draw_parameters_bundle["if_show_right_cluster"]
    if_show_left_cluster  = draw_parameters_bundle["if_show_left_cluster"]
    if_show_scatters      = draw_parameters_bundle["if_show_scatters"]
    min_line_length       = draw_parameters_bundle["min_len"]  
    slope_threshold       = draw_parameters_bundle["s_threshold"]
    painting_color        = draw_parameters_bundle["color"] 
    line_channel          = draw_parameters_bundle["channel"]
    max_line_gap          = draw_parameters_bundle["max_gap"]  
    trap_height           = draw_parameters_bundle["t_height"]
    threshold             = draw_parameters_bundle["h_threshold"] 
    data_type             = draw_parameters_bundle["d_type"]  
    thickness             = draw_parameters_bundle["thick"] 
    theta                 = draw_parameters_bundle["theta"]     
    rho                   = draw_parameters_bundle["rho"]    
    
    
    # ===============================================
    # Get Lines of the Pictures
    lines = HoughLinesP(input_image,  rho, theta, 
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
    right_X  = []
    right_Y  = []
    left_X   = []
    left_Y   = []
    height   = original_image.shape[0]
    width    = original_image.shape[1]
    center_x = width  / 2
    
    
    # ===============================================
    # Get all scatters from lines
    for cur_line in lines:
        x1, y1, x2, y2 = cur_line[0]  
                
        # ===============================================
        # Calculate Slopes
        slope = (y2 - y1) / (x2 - x1)
        print(slope, x1, x2)     
        
        
        # ===============================================
        # Seperate Points
        if abs(slope) > slope_threshold:
                      

            # ===============================================
            # Seperate Points: Points to the Left
            if slope < 0 and x1 < center_x and x2 < center_x:
                left_X.append(x1)
                left_X.append(x2)
                left_Y.append(y1)
                left_Y.append(y2)
                               
                
            # ===============================================
            # Seperate Points: Points to the Right
            elif slope > 0 and x1 > center_x and x2 > center_x:
                right_X.append(x1)
                right_X.append(x2)
                right_Y.append(y1)
                right_Y.append(y2)
                    


        
        
    # ===============================================
    # Prepare for drawing
    Y1         = input_image.shape[0]
    Y2         = input_image.shape[0] * 0.6
    line_image = np.zeros((*input_image.shape, line_channel), dtype = data_type)
    
    
    # ===============================================
    # Get Crucial Points for the left section
    train_data_left           = np.column_stack((left_X, left_Y))
    left_X1_all, left_X2_all  = clusteringPoints(train_data_left,    if_show_left_cluster, Y1, Y2, height, width)


    # ===============================================
    # Get Crucial Points for the Right section
    train_data_right            = np.column_stack((right_X, right_Y))
    right_X1_all, right_X2_all  = clusteringPoints(train_data_right, if_show_right_cluster, Y1, Y2, height, width)
    
    
    # ===============================================
    # Concatenate Vectors
    X1_all = np.concatenate((left_X1_all, right_X1_all))
    X2_all = np.concatenate((left_X2_all, right_X2_all))
    
    
    # ===============================================
    # Draw Lines
    for X1, X2 in np.column_stack((X1_all, X2_all)):
        line_image = line(line_image, (int(X1),  int(Y1)), (int(X2),  int(Y2)), painting_color, thickness)


    # ===============================================
    return line_image


# ===============================================
# Function Mixing
def mixing(original_image, line_image, mixing_para_bundle):
    original_image = original_image.astype("uint8")
    # ===============================================
    mixed_picture = addWeighted(line_image, 0.8, original_image, 1, 0.) 
    
    # ===============================================
    return mixed_picture
