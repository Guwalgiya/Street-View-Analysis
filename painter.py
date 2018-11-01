# ===============================================
# Import Packages and Functions
from   cv2               import HoughLinesP, line, addWeighted
from   random            import uniform
from   operator          import itemgetter
from   sklearn.cluster   import KMeans
from   sklearn.mixture   import GMM
import numpy             as     np
import matplotlib.pyplot as     plt
import matplotlib.image  as     mpimg


# ===============================================
# Function Draw
def draw(original_image, input_image, draw_parameters_bundle):
    
    # ===============================================
    # Get Parameters
    if_show_scatters = draw_parameters_bundle["if_show_scatters"]
    min_line_length  = draw_parameters_bundle["min_len"]  
    slope_threshold  = draw_parameters_bundle["s_threshold"]
    painting_color   = draw_parameters_bundle["color"] 
    line_channel     = draw_parameters_bundle["channel"]
    max_line_gap     = draw_parameters_bundle["max_gap"]  
    trap_height      = draw_parameters_bundle["t_height"]
    threshold        = draw_parameters_bundle["h_threshold"] 
    data_type        = draw_parameters_bundle["d_type"]  
    thickness        = draw_parameters_bundle["thick"] 
    theta            = draw_parameters_bundle["theta"]     
    rho              = draw_parameters_bundle["rho"]    
    
    
    # ===============================================
    # Get Lines of the Pictures
    lines = HoughLinesP(input_image,  rho, theta, 
                        threshold,    np.array([]), 
                        minLineLength = min_line_length, 
                        maxLineGap    = max_line_gap)
        
    right_X = []
    right_Y = []
    left_X = []
    left_Y = []
    center_x = 480
    for cur_line in lines:
        x1, y1, x2, y2 = cur_line[0]  
        
        # ===============================================
        # Calculate Slopes
        if x2 - x1 == 0.:  
            slope = -np.log(0)
        else:
            slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > slope_threshold:
            if   slope > 0 and x1 > center_x and x2 > center_x:
                right_X.append(x1)
                right_X.append(x2)
                right_Y.append(y1)
                right_Y.append(y2)
                
                
                
            elif slope < 0 and x1 < center_x and x2 < center_x:
                left_X.append(x1)
                left_X.append(x2)
                left_Y.append(y1)
                left_Y.append(y2)

    train_data = np.column_stack((left_X, left_Y))
    num_clus = 3
    cluster = GMM(n_components = num_clus, covariance_type = "full")
    labels  = cluster.fit_predict(train_data)
    d = {}
    draw_Y1 = input_image.shape[0]
    draw_Y2 = input_image.shape[0] * (1 - trap_height)
    plt.figure()
    plt.axis([0, 480 * 2, 0, 540])
    plt.scatter(left_X, left_Y, c = labels, cmap= "viridis")
    plt.gca().invert_yaxis()
    line_image = np.zeros((*input_image.shape, line_channel), dtype = data_type) 
    for i in range(num_clus):
        indices  = np.where(labels == i)[0].tolist()
        print(indices)
        if len(indices) > 1:
            reg_data = train_data[indices]
            print(reg_data)
            k, b     = np.polyfit(reg_data[:, 0], reg_data[:, 1], 1)
            d[k]     = reg_data
            draw_X1 = (draw_Y1 - b) / k
            draw_X2 = (draw_Y2 - b) / k
            line_image = line(line_image, (int(draw_X1),  int(draw_Y1)), (int(draw_X2),  int(draw_Y2)), painting_color, thickness)
        
    train_data = np.column_stack((right_X, right_Y))
    num_clus = 1
    cluster = GMM(n_components = num_clus, covariance_type = "full")
    labels  = cluster.fit_predict(train_data)
    d = {}
    draw_Y1 = input_image.shape[0]
    draw_Y2 = input_image.shape[0] * (1 - trap_height)
    plt.figure()
    plt.axis([0, 480 * 2, 0, 540])
    plt.scatter(right_X, right_Y, c = labels, cmap= "viridis")
    plt.gca().invert_yaxis()
    for i in range(num_clus):
        indices  = np.where(labels == i)[0].tolist()
        print(indices)
        if len(indices) > 1:
            reg_data = train_data[indices]
            print(reg_data)
            k, b     = np.polyfit(reg_data[:, 0], reg_data[:, 1], 1)
            d[k]     = reg_data
            draw_X1 = (draw_Y1 - b) / k
            draw_X2 = (draw_Y2 - b) / k
            line_image = line(line_image, (int(draw_X1),  int(draw_Y1)), (int(draw_X2),  int(draw_Y2)), painting_color, thickness)
 

    # ===============================================
    # Draw those scatters
    if if_show_scatters:
        temp_image = original_image.copy()
        plt.figure()
        plt.imshow(temp_image)
        plt.scatter(left_X + right_X, left_Y + right_Y)

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


# ===============================================
# Check if the current k is too much
def duplicate_or_not(k, k_list):
    if k_list == []:
        return True
    for existed_k in k_list:
        if abs(k - existed_k) < 0.2:
            return False
    return True
