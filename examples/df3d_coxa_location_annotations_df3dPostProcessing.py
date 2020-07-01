import os.path
import importlib.util
import pickle
import glob

import numpy as np

import utils_video
import utils_video.utils
import utils_video.generators

import deepfly.plot_util
from deepfly.procrustes import procrustes_seperate

from df3dPostProcessing import df3dPostProcess, convert_to_df3d_output_format


spec = importlib.util.spec_from_file_location(
    "module.name", "/home/aymanns/BRSA/examples/annotations.py"
)
annotations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotations)


def get_most_recent_pose_result(folder):
    possible_pose_results = glob.glob(os.path.join(directory, "pose_result*.pkl"))
    change_times = [os.stat(path).st_mtime for path in possible_pose_results]
    try:
        most_recent_pose_result = possible_pose_results[np.argmax(change_times)]
    except ValueError:
        print("skipped because df3d output is missing")
        most_recent_pose_result = None
    return most_recent_pose_result




def get_processed_points(directory):
    most_recent_pose_result = get_most_recent_pose_result(directory)
    if most_recent_pose_result is None:
        return None
    df3dPost = df3dPostProcess(most_recent_pose_result)
    aligned = df3dPost.align_3d_data()
    points3D = convert_to_df3d_output_format(aligned)
    points3D = points3D[:40]
    return points3D


my_points = []
clc_points = []
sg_points = []

for beh, data in annotations.annotations["my annotations"].items():
    for exp in data:
        directory = exp["directory"]
        directory = os.path.join(directory, "behData/images/df3d/")
        print(directory)
        points3D = get_processed_points(directory)
        if points3D is None:
            continue
        my_points.append(points3D)


for beh, data in annotations.annotations["Chin-Lin's data"].items():
    for exp in data:
        directory = exp["directory"]
        directory = os.path.join(directory, "images/df3d_2/")
        print(directory)
        points3D = get_processed_points(directory)
        if points3D is None:
            continue
        clc_points.append(points3D)


for beh, data in annotations.annotations["semih's annotations"].items():
    for exp in data:
        key = exp["key"]
        key = key[len("pose_result__data_paper_") :]
        fly_index = key.find("Fly")
        directory = f"/mnt/internal_hdd/aymanns/df3d_paper_annotated_experiments/{key[:fly_index-1]}/{key[fly_index:fly_index+4]}/{key[fly_index+5:fly_index+12]}/behData/images/df3d"
        print(directory)
        points3D = get_processed_points(directory)
        if points3D is None:
            continue
        sg_points.append(points3D)

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
    projected_coxa_points = np.transpose(
        np.dot(np.transpose(U), np.transpose(coxa_points))
    )
    projected_coxa_points = projected_coxa_points.reshape(
        [n_exp, -1, len(coxa_indices), 3]
    )
    for frame_idx in range(projected_coxa_points.shape[1]):
        yield utils_video.utils.plot_coxa_positions(
            projected_coxa_points[:, frame_idx], mins, maxs, labels
        )


my_generator = coxa_locations(np.array(my_points), U, mins, maxs, labels=None)
clc_generator = coxa_locations(np.array(clc_points), U, mins, maxs, labels=None)
sg_generator = coxa_locations(np.array(sg_points), U, mins, maxs, labels=None)
generator = utils_video.generators.stack([my_generator, clc_generator, sg_generator])
utils_video.make_video(
    "coxa_location_all_of_my_and_clcs_annotations.mp4", generator, fps=30
)
