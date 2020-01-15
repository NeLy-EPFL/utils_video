import itertools

import cv2
from tqdm import tqdm

from .utils import *


def make_video(video_path, frame_generator, fps, output_shape=(-1, 2880)):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    first_frame = next(frame_generator)
    frame_generator = itertools.chain([first_frame], frame_generator)
    output_shape = resize_shape(output_shape, first_frame.shape[:2])
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape[::-1])

    for img in tqdm(frame_generator):
        resized = cv2.resize(img, output_shape[::-1])
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)

    video_writer.release()
