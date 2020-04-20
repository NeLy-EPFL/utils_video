import pytest

import plotting_utils.utils

def test_resize_shape():
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((1, 2, 3), (2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3), (1, 2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2.5, 3), (2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3.4), (2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3), (2.5, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3), (2, 3.3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((-2, 3), (2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, -3), (2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3), (-2, 3))
    with pytest.raises(ValueError):
        plotting_utils.utils.resize_shape((2, 3), (2, -3))

    assert plotting_utils.utils.resize_shape((-1, -1), (2, 3)) == (2, 3)
    assert plotting_utils.utils.resize_shape((3, -1), (20, 30))[0] == 3
    assert plotting_utils.utils.resize_shape((-1, 3), (20, 30))[1] == 3
    assert plotting_utils.utils.resize_shape((20, 30), (2, 3)) == (2, 3)
    assert plotting_utils.utils.resize_shape((20, 30), (2, 3), allow_upsampling=True) == (20, 30)


def test_find_greatest_common_resolution():
    shapes = [(3, 4), (2, 6), (6, 4)]
    assert plotting_utils.utils.match_greatest_resolution(shapes, axis=0) == [(6, 8), (6, 18), (6, 4)]
    assert plotting_utils.utils.match_greatest_resolution(shapes, axis=1) == [(4, 6), (2, 6), (9, 6)]


def test_grid_size():
    with pytest.raises(ValueError):
        plotting_utils.utils.grid_size(4.5, (3, 4))
    with pytest.raises(ValueError):
        plotting_utils.utils.grid_size(4, (3,))
    with pytest.raises(ValueError):
        plotting_utils.utils.grid_size(4, (3, 4, 5))
    
    assert plotting_utils.utils.grid_size(4, (3, 4)) == (2, 2)
