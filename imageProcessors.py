# ===============================================
# Import Packages and Functions
import numpy             as     np
import matplotlib.pyplot as     plt
import cv2


# ===============================================
# Function Mixing
def mixer(original_image, line_image, mixing_para_bundle):
    
    
    # ===============================================
    # Function Mixing
    original_image = original_image.astype("uint8")
    
    
    # ===============================================
    # Get Parameters
    mixer_gamma           = mixing_para_bundle["mixer_gamma"]
    line_image_weight     = mixing_para_bundle["line_image_weight"]
    if_show_final_image   = mixing_para_bundle["if_show_final_image"]
    original_image_weight = mixing_para_bundle["original_image_weight"]
    
    
    # ===============================================
    final_image = cv2.addWeighted(line_image,     line_image_weight, 
                                  original_image, original_image_weight, 
                                  mixer_gamma) 
    
    # ===============================================
    if if_show_final_image:
        plt.figure()
        plt.imshow(final_image)
    
    # ===============================================
    return final_image



# ===============================================
# Function: colorFilter
def colorFilter(input_image, colorFilter_para_bundle):
    
    
    # ===============================================
    # Load Parameters
    gamma         = colorFilter_para_bundle["gamma"]
    lower_white   = colorFilter_para_bundle["white"]["low"]
    upper_white   = colorFilter_para_bundle["white"]["high"]
    white_weight  = colorFilter_para_bundle["white"]["weight"]
    lower_yellow  = colorFilter_para_bundle["yellow"]["low"]
    upper_yellow  = colorFilter_para_bundle["yellow"]["high"]
    yellow_weight = colorFilter_para_bundle["yellow"]["weight"]
    
    
    # ===============================================
    # Parameters to array
    lower_white  = np.array(lower_white)
    upper_white  = np.array(upper_white)
    lower_yellow = np.array(lower_yellow)
    upper_yellow = np.array(upper_yellow)
    
    
    # ===============================================
    # Change Dimension  
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    
    # ===============================================
    # Create Mask
    yellow_mask = cv2.inRange(hsv_image,   lower_yellow, upper_yellow)
    white_mask  = cv2.inRange(input_image, lower_white,  upper_white)
    
    
    # ===============================================
    # Add Mask
    yellow_image = cv2.bitwise_and(input_image, input_image,   mask = yellow_mask)
    white_image  = cv2.bitwise_and(input_image, input_image,   mask = white_mask)

    
    # ===============================================
    # Combine two masked picture
    image_out = cv2.addWeighted(white_image,  white_weight, 
                                yellow_image, yellow_weight,
                                gamma)
    
    
    # ===============================================
    return image_out


# ===============================================
def imageProcessor(input_image, process_parameters_bundle):
    
    
    # ===============================================
    # Load All the Parameters
    blur_para_bundle        = process_parameters_bundle["blur"]
    canny_para_bundle       = process_parameters_bundle["canny"]
    colorFilter_para_bundle = process_parameters_bundle["filter"]


    # ===============================================
    # Run Processes
    intermediate_image = colorFilter(input_image, colorFilter_para_bundle)
    
    
    # ===============================================
    # Run Processes
    intermediate_image = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2GRAY)
    
    
    # ===============================================
    # Run Processes
    blur_kernel        = blur_para_bundle["kernel"]
    sigma_X            = blur_para_bundle["sigma_X"]
    intermediate_image = cv2.GaussianBlur(intermediate_image, blur_kernel, sigma_X)
    
    
    # ===============================================
    # Extract Edges
    low_threshold   = canny_para_bundle["l_t"]
    high_threshold  = canny_para_bundle["h_t"]
    processed_image = cv2.Canny(intermediate_image, low_threshold, high_threshold)
    
    
    # ===============================================
    return processed_image