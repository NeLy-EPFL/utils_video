import glob

import numpy as np

import plotting_utils
import plotting_utils.generators

#points3d = np.load("example_data_df3d.npy")
#print(points3d.shape)
#print(np.where(np.isnan(points3d)))

#points3d = np.load("my_df3d_example_data.npy")
#points3d = points3d.reshape((points3d.shape[0], -1, 3))
#print(points3d.shape)
#print(np.where(np.isnan(points3d)))
#exit()

#generator = plotting_utils.generators.df3d_3d_points(points3d)
#plotting_utils.make_video("df3d.mp4", generator, fps=30)

files = glob.glob("semih*.npy")
#files = glob.glob("_mnt_data_CLC*.npy")
for f in files:
    points3d = np.load(f)
    points3d = points3d.reshape((points3d.shape[0], -1, 3))
    generator = plotting_utils.generators.df3d_3d_points(points3d)
    plotting_utils.make_video(f"{f.split('.')[0]}.mp4", generator, fps=30)
