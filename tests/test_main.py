import os.path

import pytest
import numpy as np
import cv2

import utils_video

@pytest.fixture
def black_generator():
    """
    Pytest factory that returns a generator
    that yields black frames of the given width and height.
    """
    def _black_generator(width, height, n_frames, n_channels):
        def generator():
            for i in range(n_frames):
                yield np.zeros((height, width, n_channels), dtype=np.uint8)
        return generator()
    return _black_generator

@pytest.mark.parametrize("fps,n_frames,n_channels", [(30, 60, 1), (30, 65, 3)])
def test_make_video(black_generator, tmpdir, fps, n_frames, n_channels):
    output_file = os.path.join(tmpdir, "output_vid.mp4")
    generator = black_generator(40, 20, n_frames, n_channels)
    utils_video.make_video(output_file, generator, fps)
    
    cap = cv2.VideoCapture(output_file)
    assert cap.isOpened()

    # Check fps of output video
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        cv2_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        cv2_fps = cap.get(cv2.CAP_PROP_FPS)
    assert np.isclose(fps, cv2_fps)

    # Check number of frames in output video and
    # that all the frames are black
    cv2_n_frames = 0
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        assert np.allclose(frame, 0)
        cv2_n_frames += 1
    assert cv2_n_frames == n_frames


def test_ffmpeg(tmpdir):
    output = os.path.join(tmpdir, "video.mp4")
    utils_video.ffmpeg(f"-f lavfi -i testsrc=duration=10:size=1280x720:rate=30 {output}")
"""
def test_handbrake(tmpdir):
    def bar_generator(width=100, height=50, n_frames=502, n_channels=1):
        w1 = width // 3
        w2 = width - 2*w1
        def generator():
            for i in range(n_frames):
                frame = np.concatenate(
                    (
                    np.zeros((height, w1, n_channels), dtype=np.uint8),
                    255*np.ones((height, w2, n_channels), dtype=np.uint8),
                    np.zeros((height, w1, n_channels), dtype=np.uint8)
                    ), axis=1, dtype=np.uint8)
                yield frame
        return generator()

    output_file_fullsize = os.path.join(tmpdir, "output_vid_full.mp4")
    print(f"=========={output_file_fullsize}==========")
    generator = bar_generator()  # (100, 50, n_frames, n_channels)
    utils_video.make_video(output_file_fullsize, generator, 30, use_handbrake=False)
    full_size = os.path.getsize(output_file_fullsize)

    # test handbrake inside make_video
    output_file_redsize = os.path.join(tmpdir, "output_vid_red.mp4")
    generator = bar_generator()  # (100, 50, n_frames, n_channels)
    utils_video.make_video(output_file_redsize, generator, 30, use_handbrake=True)
    
    # test handbrake on its own
    output_file_redsize2 = os.path.join(tmpdir, "output_vid_red2.mp4")
    utils_video.handbrake(video_path=output_file_fullsize, output_path=output_file_redsize2)
    
    assert os.path.isfile(output_file_redsize)
    red_size = os.path.getsize(output_file_redsize)
    assert red_size < full_size
    assert os.path.isfile(output_file_redsize2)
    red2_size = os.path.getsize(output_file_redsize2)
    assert red2_size < full_size
"""
    

