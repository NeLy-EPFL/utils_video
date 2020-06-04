from pathlib import Path

import cv2
import numpy as np

import utils2p
import utils2p.synchronization

import utils_video
import utils_video.generators
import utils_video.utils

data_dir = Path(__file__).resolve().parents[1] / "data"
experiment_dir = str(data_dir / "191129_ABO/Fly2/001_coronal")
local_correlation_image_file = str(data_dir / "191129_ABO/Fly2/local_corr.tif")
roi_mask_image_file = str(data_dir / "191129_ABO/Fly2/roi_mask.tif")
dff_traces_file = str(data_dir / "191129_ABO/Fly2/001_coronal/2p/dff_roi_traces.npy")


# Generate image showing the ROI in different colours
roi_background_image = cv2.imread(local_correlation_image_file)
roi_mask_image = cv2.imread(roi_mask_image_file)

roi_image, colors_for_rois = utils_video.utils.roi_image(roi_background_image, roi_mask_image)

# Get times for synchronization
sync_file = utils2p.find_sync_file(experiment_dir)
metadata_file = utils2p.find_metadata_file(experiment_dir)
sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
#for key, value in processed_lines.items():
#    processed_lines[key] = value[:300000]
frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])

# Generate ridge_line video
dff_traces = np.load(dff_traces_file)
ridge_line_generator = utils_video.generators.ridge_line(dff_traces, frame_times_2p, frame_times_beh, 2)

roi_image_generator = utils_video.generators.static_image(roi_image, len(frame_times_beh))

generator = utils_video.generators.stack([roi_image_generator, ridge_line_generator])

utils_video.make_video("ridge.mp4", generator, 30)

