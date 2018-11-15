# ===============================================
# Import Packages and Functions
from   cv2               import fillPoly, bitwise_and, line
import numpy             as     np
import matplotlib.pyplot as     plt


# ===============================================
# Function getVertices
def scope(original_image, input_image, vertices_parameters_bundle):
    
    
    # ===============================================
    # Get Parameters
    mask_color        = vertices_parameters_bundle["m_color"]
    if_show_region    = vertices_parameters_bundle["if_show_region"]
    height, width     = input_image.shape
    trap_height       = vertices_parameters_bundle["t_height"]
    trap_top_width    = vertices_parameters_bundle["t_twidth"]
    trap_bottom_width = vertices_parameters_bundle["t_bwidth"]
    thickness         = vertices_parameters_bundle["thickness"]

    
    # ===============================================
    # Vertices    
    a = ((width * (1     - trap_bottom_width))      // 2, height)
    #b = ((width * (1     - trap_top_width))         // 2, height - height * trap_height)
    b = ((width * (1     - trap_bottom_width))      // 2, height - height * trap_height)
    c = (width  - (width * (1 - trap_top_width))    // 2, height - height * trap_height)
    d = (width  - (width * (1 - trap_bottom_width)) // 2, height)
    
    
    # ===============================================
    # Combine 
    vertices = np.array([[a, b, c, d]], dtype = np.int32)
    
    
    # ===============================================
    # if we want to show the "trap-area"
    if if_show_region:
        temp_image = original_image.copy()
        temp_iamge = line(temp_image, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), [mask_color,], thickness)
        temp_iamge = line(temp_iamge, (int(b[0]),int(b[1])), (int(c[0]),int(c[1])), [mask_color,], thickness)
        temp_iamge = line(temp_iamge, (int(c[0]),int(c[1])), (int(c[0]),int(c[1])), [mask_color,], thickness)
        temp_iamge = line(temp_iamge, (int(d[0]),int(d[1])), (int(c[0]),int(c[1])), [mask_color,], thickness)
        plt.figure()
        plt.imshow(temp_iamge)
    
    
    # ===============================================
    # Draw Region of Interest
    target_region = get_target_region(input_image, vertices, mask_color)

    
    # ===============================================
    return target_region


# ===============================================
# Function region_of_interest
def get_target_region(input_image, vertices, mask_color):
    
    
    # ===============================================
    # Single Channel or Multiple Channels
    if len(input_image.shape) > 2:
        mask_color  = (mask_color, ) * input_image.shape[2]
        
        
    # ===============================================
    # Create a Mask to Cover Useless Things 
    mask = np.zeros_like(input_image)
    mask = fillPoly(mask, vertices, mask_color)

    
    # ===============================================
    # Draw Interests of Reigion
    target_region = bitwise_and(input_image, mask)
    
    
    # ===============================================
    return target_region