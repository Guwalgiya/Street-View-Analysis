# ===============================================
# Import Packages and Functions
import numpy as np


# ===============================================
# Changeable Operating System Parameters
slash = "\\"


# ===============================================
# Changeable Database Parameters
image_folder        = "C:\\Street-View-Analysis\\Data"
output_image_name   = ""
original_image_name = "exp1.jpg"
original_image_path = image_folder + slash + original_image_name


# ===============================================
# Changeable Database Parameters
video_folder      = "C:\\Street-View-Analysis\\Data"
input_video_name  = "solidWhiteRight.mp4"
input_video_path  = video_folder + slash + input_video_name
output_video_name = "white.mp4"


# ===============================================
# Changeable Process Parameters
gamma            = 0
sigma_X          = 0
kernel_size      = 1
blur_kernel      = (kernel_size, kernel_size)
lower_white      = [155, 155, 155] #155 155 155
upper_white      = [255, 255, 255]
lower_yellow     = [90,  100, 100]
upper_yellow     = [110, 255, 255]
white_weight     = 1
yellow_weight    = 0
low_threshold    = 100
high_threshold   = 150


# ===============================================
# Target Region Parameters
mask_color        = 255 
trap_height       = 0.34    # 0.4
trap_top_width    = 0.3     # 0.7
trap_bottom_width = 1       # 0.85


# ===============================================
# Hough Transform
rho             = 3
thick           = 15
data_type       = np.uint8
threshold       = 15
draw_height     = 0.65
max_line_gap    = 5
line_channel    = 3
theta_degree    = 1
theta_radius    = theta_degree * np.pi / 180
painting_color  = (0.1, 0.1, 0.1)
min_line_length = 5     # 10
slope_threshold = 0


# ===============================================
# Arrange Mixer Parameters
mixer_gamma           = 0
line_image_weight     = 1   
original_image_weight = 1


# ===============================================
# Show needed images
if_show_scatters       = True
if_show_R_cluster      = True
if_show_L_cluster      = True
if_show_final_image    = True
if_show_target_region  = True
if_show_original_image = True


# ===============================================
# Arrange all Parameters - Ground Floor - Blur_Parameters
blur_para_bundle            = {}
blur_para_bundle["kernel"]  = blur_kernel
blur_para_bundle["sigma_X"] = sigma_X


# ===============================================
# Arrange all Parameters - Ground Floor
canny_para_bundle        = {}
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
process_parameters_bundle["blur"]   = blur_para_bundle
process_parameters_bundle["canny"]  = canny_para_bundle
process_parameters_bundle["filter"] = colorFilter_para_bundle


# ===============================================
# Arrange all Parameters - Top Level
vertices_parameters_bundle                          = {}
vertices_parameters_bundle["m_color"]               = mask_color
vertices_parameters_bundle["t_height"]              = trap_height
vertices_parameters_bundle["t_twidth"]              = trap_top_width
vertices_parameters_bundle["t_bwidth"]              = trap_bottom_width
vertices_parameters_bundle["thickness"]             = thick
vertices_parameters_bundle["if_show_target_region"] = if_show_target_region


# ===============================================
# Arrange Painter Parameters
draw_parameters_bundle                           = {}
draw_parameters_bundle["rho"]                    = rho
draw_parameters_bundle["thick"]                  = thick
draw_parameters_bundle["theta"]                  = theta_radius
draw_parameters_bundle["color"]                  = painting_color
draw_parameters_bundle["d_type"]                 = data_type 
draw_parameters_bundle["channel"]                = line_channel
draw_parameters_bundle["max_gap"]                = max_line_gap
draw_parameters_bundle["min_len"]                = min_line_length
draw_parameters_bundle["h_threshold"]            = threshold
draw_parameters_bundle["s_threshold"]            = slope_threshold
draw_parameters_bundle["draw_height"]            = draw_height
draw_parameters_bundle["if_show_scatters"]       = if_show_scatters
draw_parameters_bundle["if_show_R_cluster"]      = if_show_R_cluster
draw_parameters_bundle["if_show_L_cluster"]      = if_show_L_cluster
draw_parameters_bundle["if_show_original_image"] = if_show_original_image


# ===============================================
# Arrange Mixer Parameters
mixing_para_bundle                          = {}
mixing_para_bundle["mixer_gamma"]           = mixer_gamma
mixing_para_bundle["line_image_weight"]     = line_image_weight 
mixing_para_bundle["if_show_final_image"]   = if_show_final_image
mixing_para_bundle["original_image_weight"] = original_image_weight 