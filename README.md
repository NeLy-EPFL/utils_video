utils_video
==============

This package provides plotting utility functions to generate videos.
Some common cases for videos of 2-photon data and behaviour videos are already
implemented in the form of generators in the `generators` module.

Usage
-----
After installing the package you can import its functions in your python code
with `import utils_video`. The main function to create videos is
`utils_video.make_video(output_file_name, generator, fps)`, where `generator`
is a python generator object that yields individual frames. A collection of
common generators can be found in the `generators` module.

Installation
------------
The package can be installed with pip. To do so first clone the repository by
running the command `git clone https://github.com/NeLy-EPFL/utils_video.git`.
This will create a copy of the code base on your machine in a folder called
`utils_video`.
Next install the package with the command `pip install -e utils_video`.

If you plan to use 3D plotting functions, make sure to install DF3D following the guidelines [here](https://github.com/NeLy-EPFL/DeepFly3D).
