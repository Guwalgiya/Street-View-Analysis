# ===============================================
# Import Packages and Functions
from   imageProcessors  import mixer, imageProcessor
from   regionDetector   import scope
from   moviepy.editor   import VideoFileClip
from   lineDetector     import lineDetector
from   os               import remove
import matplotlib.image as     mpimg
import controlPanel     as     cp
import numpy            as     np


# ===============================================
# Prepare I
initial_data_array = np.array([])
initial_data_array = initial_data_array.reshape((len(initial_data_array), 2))


# ===============================================
# Prepare II
dict_empty = {}
for i in range(12):
    dict_empty[-i] = initial_data_array
np.save("train_data_L.npy", dict_empty)
np.save("train_data_R.npy", dict_empty)


# ===============================================
# Load Image
original_image = mpimg.imread(cp.original_image_path)


# ===============================================
# this function is used to work on videos
def ensemble(input_image):
    processed_image = imageProcessor(original_image, cp.process_parameters_bundle)
    target_region   = scope(original_image, processed_image, cp.vertices_parameters_bundle)
    lines_image     = lineDetector(original_image, target_region, cp.draw_parameters_bundle, "image")
    final_image     = mixer(original_image, lines_image, cp.mixing_para_bundle)
    return final_image


# =============================================== 
# Call function ENSEMBLE to perform lane detection on a short video
input_clip  = VideoFileClip(cp.input_video_path)
output_clip = input_clip.fl_image(ensemble) 
output_clip.write_videofile(cp.output_video_name, audio = False, verbose = False)


# =============================================== 
# Clean Temp File
remove("train_data_L.npy")
remove("train_data_R.npy")