# ===============================================
# Import Packages and Functions
from   cv2               import cvtColor, GaussianBlur, Canny, COLOR_BGR2GRAY
from   CVhelpers         import colorFilter


# ===============================================
def processImage(input_image, process_parameters_bundle):
    
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
    intermediate_image = cvtColor(intermediate_image, COLOR_BGR2GRAY)
    #plt.figure()
    #plt.imshow(intermediate_image)
    
    
    # ===============================================
    # Run Processes
    blur_kernel        = blur_para_bundle["kernel"]
    sigma_X            = blur_para_bundle["sigma_X"]
    intermediate_image = GaussianBlur(intermediate_image, blur_kernel, sigma_X)
    #plt.figure()
    #plt.imshow(intermediate_image)
    
    
    # ===============================================
    # Extract Edges
    low_threshold   = canny_para_bundle["l_t"]
    high_threshold  = canny_para_bundle["h_t"]
    processed_image = Canny(intermediate_image, low_threshold, high_threshold)
    #plt.figure()
    #plt.imshow(edges)
    
    
    # ===============================================
    return processed_image
    