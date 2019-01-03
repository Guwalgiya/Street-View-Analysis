# ===============================================
# Import Packages and Functions
from   lineDetector     import lineDetector
from   regionDetector   import scope
from   imageProcessors  import mixer, imageProcessor
import matplotlib.image as     mpimg
import controlPanel     as     cp


# ===============================================
# Load Image
original_image = mpimg.imread(cp.original_image_path)


# ===============================================
# Process Image
processed_image = imageProcessor(original_image, cp.process_parameters_bundle)


# ===============================================
# Find traget region
target_region = scope(original_image, processed_image, cp.vertices_parameters_bundle)


# ===============================================
# Draw Lines
lines_image = lineDetector(original_image, target_region, cp.draw_parameters_bundle, "image")


# ===============================================
# Combine
final_image = mixer(original_image, lines_image, cp.mixing_para_bundle)


