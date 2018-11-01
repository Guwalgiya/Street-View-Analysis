import matplotlib.pyplot as     plt
import matplotlib.image  as     mpimg
import numpy             as     np
from sklearn.cluster import KMeans
image_folder      = "C:\\Street-View-Analysis\\Data"
input_image_name  = "solidWhiteCurve.jpg"
input_image_path  = image_folder + "\\" + input_image_name
input_image = mpimg.imread(input_image_path)

lines = np.load("lines.npy")
x = []
y = []
slopes= []
for cur_line in lines:
    x1, y1, x2, y2 = cur_line[0]  
    
    # ===============================================
    # Calculate Slopes
    if x2 - x1 == 0.:  
        slope = -np.log(0)
    else:
        slope = (y2 - y1) / (x2 - x1)
 

    # ===============================================
    # Find Scatter 
    x.append(x1)
    x.append(x2)
    y.append(y1)
    y.append(y2)
    

