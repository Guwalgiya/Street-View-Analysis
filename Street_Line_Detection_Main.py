# ===============================================
# Import Packages and Functions
from   moviepy.editor    import VideoFileClip
from   IPython.display   import HTML
from   detectStreetLines import processImage
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy             as np
import math
import cv2


# ===============================================
# Changeable Operating System Parameters
slash = "\\"


# ===============================================
# Changeable Database Parameters
image_folder      = "C:\\Street-View-Analysis\\Data"
input_image_name  = "solidWhiteCurve.jpg"
output_image_name = ""
video_folder      = "C:\\Street-View-Analysis\\Data"
input_video_name  = "solidWhiteRight.mp4"
output_video_name = "white.mp4"


# ===============================================
# Changeable Process Parameters
blur_kernel_size = 3
sigma_X          = 0
low_threshold    = 50
high_threshold   = 150
lower_white      = [200, 200, 200]
upper_white      = [255, 255, 255]
lower_yellow     = [90,  100, 100]
upper_yellow     = [110, 255, 255]
white_weight     = 1
yellow_weight    = 1
gamma            = 0


# ===============================================
# Integrate Database Parameters
input_image_path = image_folder + slash + input_image_name


# ===============================================
# Integrate Processing Parameters 
blur_kernel = (blur_kernel_size, blur_kernel_size)


# ===============================================
# Arrange all Parameters - Ground Floor - Blur_Parameters
blur_para_bundle            = {}
blur_para_bundle["kernel"]  = blur_kernel
blur_para_bundle["sigma_X"] = sigma_X

# ===============================================
# Arrange all Parameters - Ground Floor
canny_para_bundle = {}
canny_para_bundle["l_t"] = low_threshold
canny_para_bundle["h_t"] = high_threshold


# ===============================================
# Arrange all Parameters - Ground Floor
whiteFilter_para_bundle           = {}
whiteFilter_para_bundle["low"]    = lower_white
whiteFilter_para_bundle["high"]   = upper_white
whiteFilter_para_bundle["weight"] = white_weight


# ===============================================
# Arrange all Parameters - Ground Floor
yellowFilter_para_bundle           = {}
yellowFilter_para_bundle["low"]    = lower_yellow
yellowFilter_para_bundle["high"]   = upper_yellow
yellowFilter_para_bundle["weight"] = yellow_weight


# ===============================================
# Arrange all Parameters - Second Floor
colorFilter_para_bundle           = {}
colorFilter_para_bundle["gamma"]  = gamma
colorFilter_para_bundle["white"]  = whiteFilter_para_bundle
colorFilter_para_bundle["yellow"] = yellowFilter_para_bundle


# ===============================================
# Arrange all Parameters - Top Level
process_parameters_bundle           = {}
process_parameters_bundle["filter"] = colorFilter_para_bundle
process_parameters_bundle["canny"]  = canny_para_bundle
process_parameters_bundle["blur"]   = blur_para_bundle
print(process_parameters_bundle)

# ===============================================
#input_image   = mpimg.imread(input_image_path)
#plt.imshow(input_image)
#labeled_image = processImage(input_image, process_para_bundle)


# ===============================================
# plt.imshow(input_image)
#plt.imshow(labeled_image)


# ===============================================
#video_clip = VideoFileClip(video_folder + slash + video_name_in)
#print(video_clip)
