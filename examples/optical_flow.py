import numpy as np

import utils_video
import utils_video.generators

import utils2p
import utils2p.synchronization

output_file = "opt_flow.mp4"
sync_folder = "/media/LaCie/Student_data/ball_rotation/sync_002"
imgs_dir = "/media/LaCie/Student_data/ball_rotation/behData_002/images"
optical_flow_folder = "/media/LaCie/Student_data/ball_rotation/behData_002/OptFlowData"

sync_file = utils2p.find_sync_file(sync_folder)
sync_metadata_file = utils2p.find_sync_metadata_file(sync_folder)
cam_line, opt_flow_line = utils2p.synchronization.get_lines_from_h5_file(sync_file, ["Basler", "OpFlow"])
capture_json = utils2p.find_seven_camera_metadata_file(imgs_dir)
cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
opt_flow_line = utils2p.synchronization.process_optical_flow_line(opt_flow_line)


# Get exact times of each camera frame
sync_metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
freq = sync_metadata.get_freq()
sync_times = utils2p.synchronization.get_times(len(cam_line), freq=freq)
indices = utils2p.synchronization.edges(cam_line, size=(0, np.inf))
if cam_line[0] >= 0:
    with_start_frame = np.zeros(len(indices[0]) + 1, dtype=np.int)
    with_start_frame[1:] = indices[0]
    indices = (with_start_frame,)
cam_times = sync_times[indices]

# Get optical flow data with synchronisation
optical_flow_file = utils2p.find_optical_flow_file(optical_flow_folder)
optical_flow = utils2p.load_optical_flow(optical_flow_file, 
                                         gain_0_x=round(1 / 1.04, 2),
                                         gain_0_y=round(1 / 1.01, 2),
                                         gain_1_x=round(1 / 1.01, 2),
                                         gain_1_y=round(1 / 0.98, 2),
                                         smoothing_kernel=np.ones(300) / 300,
                                        )
pitch = optical_flow["vel_pitch"]
yaw = optical_flow["vel_yaw"]
roll = optical_flow["vel_roll"]
indices = utils2p.synchronization.edges(opt_flow_line, size=(0, np.inf))
optical_flow_times = sync_times[indices]


camera_imgs_generator = utils_video.generators.video("/media/LaCie/Student_data/ball_rotation/behData_002/videos/camera_6.mp4")

window = 10
opt_flow_generator = utils_video.generators.optical_flow(pitch, yaw, roll, optical_flow_times, window, frame_times=cam_times)

frame_generator = utils_video.generators.stack([camera_imgs_generator, opt_flow_generator], axis=0)
utils_video.make_video(output_file, frame_generator, fps=100, n_frames=500)
