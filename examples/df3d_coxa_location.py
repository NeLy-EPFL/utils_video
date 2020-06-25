import glob

import numpy as np

import utils_video
import utils_video.generators

import deepfly.plot_util
from deepfly.procrustes import procrustes_seperate

path_to_processed_points = "/mnt/data/CLC/181227_R15E08-tdTomGC6fopt/Fly2/CO2xzGG/behData_001/images/df3d_2/points3D.csv"
points3d = np.genfromtxt(path_to_processed_points, delimiter=",")
points3d = points3d.reshape((points3d.shape[0], -1, 3))
points3d = points3d[:100]

points3d_rot =  deepfly.plot_util.rotate_points3d(points3d.copy())

points3d_proc_rot = procrustes_seperate(points3d.copy())
points3d_proc_rot = deepfly.plot_util.rotate_points3d(points3d_proc_rot)

points3d_rot_proc = deepfly.plot_util.rotate_points3d(points3d.copy())
points3d_rot_proc = procrustes_seperate(points3d_rot_proc)


#path_to_processed_points = "/mnt/data/CLC/181227_R15E08-tdTomGC6fopt/Fly2/CO2xzGG/behData_002/images/df3d_2/points3D.csv"
path_to_processed_points = "/mnt/data/FA/191129_ABO/Fly2/012_coronal/behData/images/df3d/points3D.csv"
points3d_2 = np.genfromtxt(path_to_processed_points, delimiter=",")
points3d_2 = points3d_2.reshape((points3d_2.shape[0], -1, 3))
points3d_2 = points3d_2[:100]

#generator_coxa_location = utils_video.generators.coxa_locations(np.array([points3d, points3d_rot, points3d_proc_rot, points3d_rot_proc, points3d_2]), labels=["raw", "rot", "proc_rot", "rot_proc", "my_points"])
generator_coxa_location = utils_video.generators.coxa_locations(np.array([points3d, points3d_2]), labels=["raw", "my_points"])

generator_exp_1 = utils_video.generators.df3d_3d_points(points3d)
generator_exp_2 = utils_video.generators.df3d_3d_points(points3d_2)
generator = utils_video.generators.stack([generator_exp_1, generator_coxa_location, generator_exp_2], axis=1)
utils_video.make_video(f"coxa_location.mp4", generator, fps=30)
