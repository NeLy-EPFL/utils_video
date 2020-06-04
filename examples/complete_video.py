import numpy as np

import utils2p
import utils2p.synchronization

import utils_video
import utils_video.generators

experiment_dir = "/mnt/data/FA/191129_ABO/Fly2/001_coronal"
local_correlation_image_file = "/mnt/data/FA/191129_ABO/Fly2/local_corr.tif"
roi_mask_image_file = "/mnt/data/FA/191129_ABO/Fly2/roi_mask.tif"
dff_traces_file = "/mnt/data/FA/191129_ABO/Fly2/001_coronal/2p/dff_roi_traces.npy"
dff_stack_file = "/mnt/data/FA/191129_ABO/Fly2/001_coronal/2p/dff.tif"
beh_images = "/mnt/data/FA/191129_ABO/Fly2/001_coronal/behData/images/camera_1_img_*.jpg"
pca_points_file = "/mnt/data/FA/191129_ABO/Fly2/001_coronal/2p/pca_dff_traces.npy"

def crop(img):
    start_0, stop_0 = (80, 410)
    start_1, stop_1 = (70, 620)
    if img.ndim == 4:
        return img[:, start_0 : stop_0, start_1 : stop_1, :]
    elif img.ndim == 3 and img.shape[2] in [3, 4]:
        return img[start_0 : stop_0, start_1 : stop_1, :]
    elif img.ndim == 3:
        return img[:, start_0 : stop_0, start_1 : stop_1]
    elif img.ndim == 2:
        return img[start_0 : stop_0, start_1 : stop_1]
    raise ValueError("img must have 2, 3 or 4 dimensions")

# Get times for synchronization
sync_file = utils2p.find_sync_file(experiment_dir)
metadata_file = utils2p.find_metadata_file(experiment_dir)
sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])

beh_generator = utils_video.generators.images(beh_images)

# Add time stamp
text = [f"{t:.1f} s" for t in frame_times_beh]
beh_generator = utils_video.generators.add_text(beh_generator, text, scale=3, pos=(680, 100))


pca_points = np.load(pca_points_file)
dynamics_3D_generator = utils_video.generators.dynamics_3D(pca_points, 10, fig_size=(6, 3))
#dynamics_3D_generator = utils_video.generators.pad(dynamics_3D_generator, 50, 0, 0, 0)


dff_stack = utils2p.load_img(dff_stack_file)

dff_stack = crop(dff_stack)

dff_generator = utils_video.generators.dff(dff_stack)
indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), processed_lines["Cameras"], processed_lines["Frame Counter"])
dff_generator = utils_video.generators.resample(dff_generator, indices)
dff_generator = utils_video.generators.pad(dff_generator, 50, 0, 0, 0)



# Generate image showing the ROI in different colours
roi_background_image = utils2p.load_img(local_correlation_image_file)
roi_mask_image = utils2p.load_img(roi_mask_image_file)

roi_background_image = crop(roi_background_image)
roi_mask_image = crop(roi_mask_image)

roi_image, colors_for_rois = utils_video.utils.roi_image(roi_background_image, roi_mask_image)

# Generate ridge_line video
dff_traces = np.load(dff_traces_file)
ridge_line_generator = utils_video.generators.ridge_line(dff_traces, frame_times_2p, frame_times_beh, 2)

roi_image_generator = utils_video.generators.static_image(roi_image, len(frame_times_beh))
roi_image_generator = utils_video.generators.pad(roi_image_generator, 0, 50, 50, 50)

left_hand_side_generator = utils_video.generators.stack([roi_image_generator, ridge_line_generator])
left_hand_side_generator = utils_video.generators.pad(left_hand_side_generator, 0, 0, 50, 0)
right_hand_side_generator = utils_video.generators.stack([beh_generator, dynamics_3D_generator, dff_generator])

generator = utils_video.generators.stack([left_hand_side_generator, right_hand_side_generator], axis=1)
generator = utils_video.generators.pad(generator, 50, 50, 50, 50)
utils_video.make_video("complete_video.mp4", generator, 30)

