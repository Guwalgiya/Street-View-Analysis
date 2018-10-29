# ===============================================
# Import Packages and Functions
from   cv2               import cvtColor, GaussianBlur, Canny, COLOR_BGR2GRAY
from   CVhelpers         import colorFilter
import matplotlib.pyplot as     plt

# ===============================================
def processImage(input_image, process_para_bundle):
    
    # ===============================================
    # Load All the Parameters
    [blur_para_bundle, colorFilter_para_bundle] = process_para_bundle
    [blur_kernel,      sigma_X]                 = blur_para_bundle
    
    # ===============================================
    # Run Processes
    intermediate_image = colorFilter(input_image, colorFilter_para_bundle)
    plt.figure()
    plt.imshow(intermediate_image)
    
    intermediate_image = cvtColor(intermediate_image, COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(intermediate_image)
    
    intermediate_image = GaussianBlur(intermediate_image, blur_kernel, sigma_X)
    plt.figure()
    plt.imshow(intermediate_image)
    
    
    # ===============================================
    # Extract Edges
    edges = Canny(intermediate_image, 50, 150)
    plt.figure()
    plt.imshow(edges)
    
    
    
    return 0
    