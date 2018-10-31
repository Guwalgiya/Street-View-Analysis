# ===============================================
# Import Packages and Functions
from   cv2               import HoughLinesP, line, addWeighted
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
        

    # ===============================================
    # Get Slopes of Lines
    x            = []
    y            = []
    slopes       = []
    chosen_lines = []
    for cur_line in lines:
        x1, y1, x2, y2 = cur_line[0]  
        
        # ===============================================
        # Calculate Slopes
        if x2 - x1 == 0.:  
            slope = -np.log(0)
        else:
            slope = (y2 - y1) / (x2 - x1)
 
    
        # ===============================================
        # Find Scatter 
        x.append(x1)
        x.append(x2)
        y.append(y1)
        y.append(y2)
        
        
        # ===============================================
        # Choose Qualified Lines
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            chosen_lines.append(cur_line)
    
    
    # ===============================================
    # Draw those scatters
    if if_show_scatters:
        temp_image = original_image.copy()
        plt.figure()
        plt.imshow(temp_image)
        plt.scatter(x, y)
    
    
    
    # ===============================================
    # Get Lines   
    right_lines = []
    left_lines  = []
    center_x    = input_image.shape[1] / 2  
    for i, cur_line in enumerate(chosen_lines):
        # ===============================================
        # Put them into correct class
        x1, y1, x2, y2 = cur_line[0]    
        if   slopes[i] > 0 and x1 > center_x and x2 > center_x:
            right_lines.append(cur_line)
        elif slopes[i] < 0 and x1 < center_x and x2 < center_x:
            left_lines.append(cur_line)

       
    # ===============================================
    # Prepare to Draw
    Y1 = input_image.shape[0]
    Y2 = input_image.shape[0] * (1 - trap_height)
    
    
    # ===============================================
    # Work on Left Lines
    left_lines_x = []
    left_lines_y = []
    
    for cur_line in left_lines:
        x1, y1, x2, y2 = cur_line[0]
        
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        
        left_lines_y.append(y1)
        left_lines_y.append(y2)
        
    if len(left_lines_x) > 0:
        left_k, left_b = np.polyfit(left_lines_x, left_lines_y, 1)
    else:
        left_k, left_b = 1, 1
    
    left_x1 = (Y1 - left_b) / left_k
    left_x2 = (Y2 - left_b) / left_k

    
    # ===============================================
    # Work on Right Lines
    right_lines_x = []
    right_lines_y = []
    
    for cur_line in right_lines:
        x1, y1, x2, y2 = cur_line[0]
        
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    if len(right_lines_x) > 0:
        right_k, right_b = np.polyfit(right_lines_x, right_lines_y, 1)
    else:
        right_k, right_b = 1, 1
    
    right_x1 = (Y1 - right_b) / right_k
    right_x2 = (Y2 - right_b) / right_k  
    

    # ===============================================
    # Draw        
    line_image = np.zeros((*input_image.shape, line_channel), dtype = data_type) 
    line_image = line(line_image, (int(left_x1),  int(Y1)), (int(left_x2),  int(Y2)), painting_color, thickness)
    line_image = line(line_image, (int(right_x1), int(Y1)), (int(right_x2), int(Y2)), painting_color, thickness)
    
    
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