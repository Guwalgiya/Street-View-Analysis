import matplotlib.pyplot as     plt
import matplotlib.image  as     mpimg
import numpy             as     np
image_folder      = "C:\\Street-View-Analysis\\Data"
input_image_name  = "solidWhiteCurve.jpg"
input_image_path  = image_folder + "\\" + input_image_name
input_image = mpimg.imread(input_image_path)

lines = np.load("lines.npy")
x = []
y = []
for i in range(len(lines)):
    if i == 6:
        line = lines[i][0]
        print(line)
        x.append(line[0])
        x.append(line[2])
        y.append(line[1])
        y.append(line[3])
        plt.figure()
        plt.imshow(input_image)
        plt.scatter(x, y)
        x = []
        y = []
