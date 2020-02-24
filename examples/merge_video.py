import os.path
import glob

import plotting_utils
import plotting_utils.generators

folder = "brsa_test"
paths = glob.glob(os.path.join(folder, "*.mp4"))

generator = plotting_utils.generators.merge_videos(paths)

output_file = os.path.join(folder, "merged.mp4")
plotting_utils.make_video(output_file, generator, 30)
