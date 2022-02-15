import itertools
import subprocess
import os

import cv2
from tqdm import tqdm

from .utils import resize_shape


def make_video(video_path, frame_generator, fps, output_shape=(-1, 2880), n_frames=-1, use_handbrake=False):
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
    use_handbrake : bool
        If true, attempt to use handbrake to compress the video. Default: False
    """
    if float(fps).is_integer() and int(fps) != 1 and (int(fps) & (int(fps) - 1)) == 0:
        import warnings

        warnings.warn(
            f"Frame rate {fps} is a power of 2. This can result in faulty video files."
        )
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

    if use_handbrake:
        handbrake(video_path)

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

def handbrake(video_path, output_path=None):
    """apply HandBrake to a video to compress it.
    This is an EXPERIMENTAL FEATURE and requires that the Handbrake command line
    interface is installed!

    For installation follow the following steps:
    Download the command line interface (CLI) from here:
    https://handbrake.fr/downloads2.php
    using the following instructions:
    https://handbrake.fr/docs/en/1.5.0/get-handbrake/download-and-install.html
    >>> flatpak --user install HandBrakeCLI-1.4.2-x86_64.flatpak
    You might have to install flatpak with apt-get first.
    If error about unacceptable TLS certificate pops up: 
    >>> sudo apt install --reinstall ca-certificates
    Add flatpak to your PATH. This way you can use the commands below
    >>> export PATH=$PATH:$HOME/.local/share/flatpak/exports/bin:/var/lib/flatpak/exports/bin
    Now the CLI can be run as follows:
    https://handbrake.fr/docs/en/latest/cli/cli-options.html
    >>> fr.handbrake.HandBrakeCLI -i source -o destination

    Parameters
    ----------
    video_path : str
        path to your .mp4 video file
    output_path : str, optional
        where the compressed video should be saved to. If None, overwrite original video, by default None
    """
    if output_path is None:
        REPLACE = True
        folder, file_name = os.path.split(video_path)
        output_path = os.path.join(folder, "tmp_" + file_name)
    else:
        REPLACE = False
    
    export_path = "export PATH=$PATH:$HOME/.local/share/flatpak/exports/bin:/var/lib/flatpak/exports/bin"
    # check whether handbrake CLI is installed
    if os.system(export_path+" && fr.handbrake.HandBrakeCLI -h >/dev/null 2>&1"):
        print("HandBrakeCLI is not installed.\n",
              "Install files can be found here: https://handbrake.fr/downloads2.php \n",
              "Install instructions here: https://handbrake.fr/docs/en/1.5.0/get-handbrake/download-and-install.html")
        return
    # run the client on the video
    os.system(export_path+f" && fr.handbrake.HandBrakeCLI -i {video_path} -o {output_path}")
    if REPLACE:
        os.system(f"mv {output_path} {video_path}")