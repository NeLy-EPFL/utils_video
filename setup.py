from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="utils_video",
    version="0.1",
    packages=["utils_video",],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Basic utility functions for plotting videos with 2p data and behaviour.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/utils_video.git",
    install_requires=["pytest", "numpy", "matplotlib", "opencv-python", "tqdm"],
    entry_points={"console_scripts": ["compress_video = utils_video.compress_video"]},
)
