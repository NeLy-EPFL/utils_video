import os.path

import numpy as np

import utils_video
import utils_video.generators


folder = "/mnt/data/CLC/181227_R15E08-tdTomGC6fopt/Fly2/CO2xzGG/behData_001/images/"
path_to_change_points = os.path.join(
    folder, "df3d_2/divisive_estimates_pca_of_3d_9.csv"
)
change_points = np.genfromtxt(path_to_change_points, delimiter=",", skip_header=1)[
    :, 1
].astype(np.int)
print(change_points)

generator = utils_video.generators.images(os.path.join(folder, "camera_6_img_*.jpg"))
generator = utils_video.generators.change_points(generator, change_points, n_pause=30)

utils_video.make_video("change_points_vid.mp4", generator, 30)
