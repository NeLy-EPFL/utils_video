import itertools
import subprocess

import cv2
from tqdm import tqdm

from .utils import resize_shape


def make_video(video_path, frame_generator, fps, output_shape=(-1, 2880), n_frames=-1):
    """
    This function writes a video to file with all frames that
    the `frame_generator` yields.

    Parameters
    ----------
    video_path : string
        Name/path to the output file.
    frame_generator : generator
        Generator yielding individual frames.
    fps : int
        Frame rate in frames per second.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_frame = next(frame_generator)
    frame_generator = itertools.chain([first_frame], frame_generator)
    output_shape = resize_shape(output_shape, first_frame.shape[:2])
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape[::-1])

    for frame, img in tqdm(enumerate(frame_generator)):
        resized = cv2.resize(img, output_shape[::-1])
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)
        if frame == n_frames - 1:
            break

    video_writer.release()


def ffmpeg(command, pixel_format="yuv420p"):
    """
    Interface to run ffmpeg from python.

    Parameters
    ----------
    command : string
        command passed to ffmpeg.
    pixel_format : string
        Specifies the pixel format of the output video.
        If `command` includes '-pix_fmt', `pixel_format
        is ignored. Default is 'yuv420p' which ensures
        videos can be included in presentation.
    """
    command_list = command.split()
    if not "-pix_fmt" in command_list:
        command_list = (
            command_list[:-1] + ["-pix_fmt", pixel_format] + [command_list[-1],]
        )
    subprocess.run(["ffmpeg",] + command_list, check=True)
