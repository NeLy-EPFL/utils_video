import os.path
import glob

import utils_video
import utils_video.generators

folder = "brsa_test"
paths = glob.glob(os.path.join(folder, "*.mp4"))

generator = utils_video.generators.merge_videos(paths)

output_file = os.path.join(folder, "merged.mp4")
utils_video.make_video(output_file, generator, 30)
