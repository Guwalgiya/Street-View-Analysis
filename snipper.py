# ===============================================
# Import Packages and Functions
from   cv2   import fillPoly, bitwise_and
import numpy as     np


# ===============================================
# Function getVertices
def scope(input_image, vertices_parameters_bundle):
    
    
    # ===============================================
    # Get Parameters
    mask_color        = vertices_parameters_bundle["m_color"]
    height, width     = input_image.shape
    trap_height       = vertices_parameters_bundle["t_height"]
    trap_top_width    = vertices_parameters_bundle["t_twidth"]
    trap_bottom_width = vertices_parameters_bundle["t_bwidth"]
    
    
    # ===============================================
    # Vertices    
    a = ((width * (1     - trap_bottom_width))      // 2, height)
    b = ((width * (1     - trap_top_width))         // 2, height - height * trap_height)
    c = (width  - (width * (1 - trap_top_width))    // 2, height - height * trap_height)
    d = (width  - (width * (1 - trap_bottom_width)) // 2, height)
    
    
    # ===============================================
    # Combine 
    vertices = np.array([[a, b, c, d]], dtype = np.int32)
    
    
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