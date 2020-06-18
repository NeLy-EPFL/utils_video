import numpy as np

import utils2p
import utils2p.synchronization

import utils_video
import utils_video.generators

experiment_dir = "/mnt/data/FA/191129_ABO/Fly2/001_coronal"

# Get times for synchronization
sync_file = utils2p.find_sync_file(experiment_dir)
metadata_file = utils2p.find_metadata_file(experiment_dir)
sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])

beh_generator = utils_video.generators.images("/mnt/data/FA/191129_ABO/Fly2/001_coronal/behData/images/camera_1_img_*.jpg")

# Add time stamp
text = [f"{t:.1f} s" for t in frame_times_beh]
beh_generator = utils_video.generators.add_text(beh_generator, text, scale=3, pos=(680, 100))

dff_stack = utils2p.load_img("/mnt/data/FA/191129_ABO/Fly2/001_coronal/2p/dff.tif")
dff_generator = utils_video.generators.dff(dff_stack)
indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), processed_lines["Cameras"], processed_lines["Frame Counter"])
dff_generator = utils_video.generators.resample(dff_generator, indices)

#generator = beh_generator
#generator = dff_generator
generator = utils_video.generators.stack([beh_generator, dff_generator], axis=1)
utils_video.make_video("beh_and_dff.mp4", generator, 30)

