import os.path

import numpy as np

import utils2p
import utils2p.synchronization

import utils_video
import utils_video.generators


experiment_dir = "/mnt/data/FA/200831_G23xU1/Fly2/001_coronal/"
path_images = os.path.join(experiment_dir, "behData/images/camera_1_img_*.jpg")
path_beh_video = os.path.join(experiment_dir, "behData/images/camera_1.mp4")
path_red_stack = os.path.join(experiment_dir, "2p/red.tif")
path_green_stack = os.path.join(experiment_dir, "2p/green.tif")

# Get times for synchronization
sync_file = utils2p.find_sync_file(experiment_dir)
metadata_file = utils2p.find_metadata_file(experiment_dir)
sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])

#beh_generator = utils_video.generators.images(path_images)
beh_generator = utils_video.generators.video(path_beh_video)

# Add time stamp
text = [f"{t:.1f}s" for t in frame_times_beh]
beh_generator = utils_video.generators.add_text(beh_generator, text, scale=3, pos=(650, 100))



red_stack = utils2p.load_img(path_red_stack)
green_stack = utils2p.load_img(path_green_stack)
raw_generator = utils_video.generators.frames_2p(red_stack, green_stack, percentiles=(2, 98))
indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), processed_lines["Cameras"], processed_lines["Frame Counter"])
raw_generator = utils_video.generators.resample(raw_generator, indices)

generator = utils_video.generators.stack([beh_generator, raw_generator], axis=1)
utils_video.make_video("beh_and_raw.mp4", generator, 100)
