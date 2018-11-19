# ===============================================
# Import Packages and Functions
from   cv2               import HoughLinesP, line, addWeighted, putText, LINE_AA
from   machine_learning  import clusteringPoints
import numpy             as     np
import matplotlib.pyplot as     plt


# ===============================================
# Function Draw
def draw(original_image, input_image, draw_parameters_bundle):
    
    
    # ===============================================
    # Get Parameters
    if_show_R_cluster = draw_parameters_bundle["if_show_R_cluster"]
    if_show_L_cluster = draw_parameters_bundle["if_show_L_cluster"]
    if_show_scatters  = draw_parameters_bundle["if_show_scatters"]
    min_line_length   = draw_parameters_bundle["min_len"]  
    slope_threshold   = draw_parameters_bundle["s_threshold"]
    painting_color    = draw_parameters_bundle["color"] 
    drawing_height    = draw_parameters_bundle["draw_height"]
    line_channel      = draw_parameters_bundle["channel"]
    max_line_gap      = draw_parameters_bundle["max_gap"]  
    threshold         = draw_parameters_bundle["h_threshold"] 
    data_type         = draw_parameters_bundle["d_type"]  
    thickness         = draw_parameters_bundle["thick"] 
    theta             = draw_parameters_bundle["theta"]     
    rho               = draw_parameters_bundle["rho"]    
    
    
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
    R_X      = []
    R_Y      = []
    L_X      = []
    L_Y      = []


    # ===============================================
    # Frame Information 
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
    Y1         = input_image.shape[0]
    Y2         = input_image.shape[0] * drawing_height
    line_image = np.zeros((*input_image.shape, line_channel), dtype = data_type)
    
    
    # ===============================================
    # Make 2-D points
    train_data_L = np.column_stack((L_X, L_Y))
    train_data_R = np.column_stack((R_X, R_Y))
    
    
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
        line_image = line(line_image, (int(X1),  int(Y1)), (int(X2),  int(Y2)), painting_color, thickness)
    
    
    # ===============================================
    # Add number of lines to a frame
    putText(line_image, str(num_L_lines + num_R_lines), (150, 150), LINE_AA, 5, painting_color, thickness)
    
    
    # ===============================================
    return line_image


# ===============================================
# Function Mixing
def mixing(original_image, line_image, mixing_para_bundle):
    
    
    # ===============================================
    # Function Mixing
    original_image = original_image.astype("uint8")
    
    
    # ===============================================
    # Get Parameters
    line_image_weight     = mixing_para_bundle["line_image_weight"]
    original_image_weight = mixing_para_bundle["original_image_weight"]
    mixer_gamma           = mixing_para_bundle["mixer_gamma"]
    
    
    # ===============================================
    mixed_picture = addWeighted(line_image_weight,     1, 
                                original_image_weight, 1, mixer_gamma) 
    
    
    # ===============================================
    return mixed_picture
