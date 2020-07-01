import os.path
import importlib.util
import pickle

import numpy as np

import utils_video
import utils_video.utils
import utils_video.generators

import deepfly.plot_util
from deepfly.procrustes import procrustes_seperate

from df3dPostProcessing import df3dPostProcess

spec = importlib.util.spec_from_file_location(
    "module.name", "/home/aymanns/BRSA/examples/annotations.py"
)
annotations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotations)

my_points = []
clc_points = []
sg_points = []

for beh, data in annotations.annotations["my annotations"].items():
    for exp in data:
        directory = exp["directory"]
        path_to_processed_points = os.path.join(
            directory, "behData/images/df3d/points3D.csv"
        )
        points3D = np.genfromtxt(path_to_processed_points, delimiter=",")
        points3D = points3D.reshape((points3D.shape[0], -1, 3))
        my_points.append(points3D[:10])


for beh, data in annotations.annotations["Chin-Lin's data"].items():
    for exp in data:
        directory = exp["directory"]
        path_to_processed_points = os.path.join(directory, "images/df3d_2/points3D.csv")
        points3D = np.genfromtxt(path_to_processed_points, delimiter=",")
        points3D = points3D.reshape((points3D.shape[0], -1, 3))
        clc_points.append(points3D[:10])


with open("/home/aymanns/BRSA/examples/pose_result_smooth.pkl", "rb") as f:
    semihs_data = pickle.load(f)

for beh, data in annotations.annotations["semih's annotations"].items():
    for exp in data:
        key = exp["key"]
        points3D = semihs_data[key]
        # points3D = procrustes_seperate(points3D.copy())
        # points3D = deepfly.plot_util.rotate_points3d(points3D)
        sg_points.append(points3D[:10])

# print(np.array(points).shape)
# exit()

points = np.array(my_points + clc_points + sg_points)
coxa_indices = np.array([0, 5, 10, 19, 24, 29])
coxa_points = points[:, :, coxa_indices, :]
coxa_points = coxa_points.reshape((-1, 3))
centroid = np.mean(coxa_points, axis=0)
coxa_points = coxa_points - centroid
U, S, VT = np.linalg.svd(np.transpose(coxa_points))
projected_coxa_points = np.transpose(np.dot(np.transpose(U), np.transpose(coxa_points)))
mins = np.min(projected_coxa_points, axis=0)
maxs = np.max(projected_coxa_points, axis=0)


def coxa_locations(points3d, U, mins, maxs, labels=None):
    # allow for multiple experiments to be shown
    if points3d.ndim == 3:
        points3d = points3d[
            np.newaxis,
        ]

    n_exp = points3d.shape[0]

    coxa_indices = np.array([0, 5, 10, 19, 24, 29])
    coxa_points = points3d[:, :, coxa_indices, :]
    coxa_points = coxa_points.reshape((-1, 3))
    centroid = np.mean(coxa_points, axis=0)
    coxa_points = coxa_points - centroid
    # U, S, VT = np.linalg.svd(np.transpose(coxa_points))
    # print("U:", U)
    projected_coxa_points = np.transpose(
        np.dot(np.transpose(U), np.transpose(coxa_points))
    )
    # mins = np.min(projected_coxa_points, axis=0)
    # maxs = np.max(projected_coxa_points, axis=0)
    projected_coxa_points = projected_coxa_points.reshape(
        [n_exp, -1, len(coxa_indices), 3]
    )
    for frame_idx in range(projected_coxa_points.shape[1]):
        yield utils_video.utils.plot_coxa_positions(
            projected_coxa_points[:, frame_idx], mins, maxs, labels
        )


# generator = coxa_locations(np.array(points), labels=None)
my_generator = coxa_locations(np.array(my_points), U, mins, maxs, labels=None)
clc_generator = coxa_locations(np.array(clc_points), U, mins, maxs, labels=None)
sg_generator = coxa_locations(np.array(sg_points), U, mins, maxs, labels=None)
generator = utils_video.generators.stack([my_generator, clc_generator, sg_generator])
utils_video.make_video(
    "coxa_location_all_of_my_and_clcs_annotations.mp4", generator, fps=30
)
