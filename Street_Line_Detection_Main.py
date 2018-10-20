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
input_image_name  = "Atlanta_2.jpg"
output_image_name = ""
video_folder      = "C:\\Street-View-Analysis\\Data"
input_video_name  = "solidWhiteRight.mp4"
output_video_name = "white.mp4"


# ===============================================
# Changeable Process Parameters
blur_kernel_size = 3
sigma_X          = 0

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
blur_kernel              = (blur_kernel_size, blur_kernel_size)
blur_para_bundle         = [blur_kernel,      sigma_X]
whiteFilter_para_bundle  = [lower_white,      upper_white,             white_weight] 
yellowFilter_para_bundle = [lower_yellow,     upper_yellow,            yellow_weight]
colorFilter_para_bundle  = [gamma,            whiteFilter_para_bundle, yellowFilter_para_bundle]  
process_para_bundle      = [blur_para_bundle, colorFilter_para_bundle]
print(process_para_bundle)


# ===============================================
input_image   = mpimg.imread(input_image_path)
labeled_image = processImage(input_image, process_para_bundle)


# ===============================================
# plt.imshow(input_image)
#plt.imshow(labeled_image)


# ===============================================
#video_clip = VideoFileClip(video_folder + slash + video_name_in)
#print(video_clip)
