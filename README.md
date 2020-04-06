plotting_utils
==============

This package provides plotting utility functions to generate videos.
Some common cases for videos of 2-photon data and behaviour videos are already
implemented in the form of generators in the `generators` module.

Usage
-----
After installing the package you can import its functions in your python code
with `import plotting_utils`. The main function to create videos is 
`plotting_utils.make_video(output_file_name, generator, fps)`, where `generator`
is a python generator object that yields individual frames. A collection of
common generators can be found in the `generators` module.

Installation
------------
The package can be installed with pip. To do so first clone the repository by
running the command `git clone https://github.com/NeLy-EPFL/plotting_utils.git`.
This will create a copy of the code base on your machine in a folder called 
`plotting_utils`.
Next install the package with the command `pip install -e plotting_utils`.
