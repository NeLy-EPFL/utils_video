import pytest
import numpy as np

import plotting_utils.generators

@pytest.fixture
def homogeneous_generator():
    """
    This is a pytest factory that returns a
    generator yielding homogeneous images of a
    given size and brightness.
    """
    def factory(size=(480, 736), vrange=range(256)):
        def generator():
            for i in vrange:
                yield np.ones(size) * i
        return generator()
    return factory


@pytest.mark.parametrize("size_gen1,size_gen2", (((480, 736), (480, 736)), ((480, 736), (400, 700))))
def test_stack(tmpdir, homogeneous_generator, size_gen1, size_gen2):
    generator1 = homogeneous_generator(size=size_gen1, vrange=range(2))
    generator2 = homogeneous_generator(size=size_gen2, vrange=range(1, -1, -1))
    stacked_generator = plotting_utils.generators.stack([generator1, generator2])

    size_gen1, size_gen2 = plotting_utils.utils.match_greatest_resolution([size_gen1, size_gen2], axis=1)
    
    first_frame = np.zeros((size_gen1[0] + size_gen2[0], size_gen1[1]))
    first_frame[size_gen1[0]:] = 1
    second_frame = np.zeros((size_gen1[0] + size_gen2[0], size_gen1[1]))
    second_frame[:size_gen1[0]] = 1

    assert np.allclose(next(stacked_generator), first_frame)
    assert np.allclose(next(stacked_generator), second_frame)
    
