from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="plotting_utils",
    version="0.1",
    packages=["plotting_utils",],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Basic utility functions for plotting videos with 2p data and behaviour.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/NeLy-EPFL/utils2p",
    install_requires=["pytest", "numpy", "matplotlib", "opencv-python"],
)
