# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without needing to import them (pytest will automatically discover them)
# https://docs.pytest.org/en/7.1.x/reference/fixtures.html#:~:text=The%20conftest.py%20file%20serves%20as%20a%20means%20of,to%20import%20them%20%28pytest%20will%20automatically%20discover%20them%29.

import numpy as np
import RedLionfishDeconv.helperfunctions as rlh
import pytest
import scipy.signal

@pytest.fixture
def data_gauss_w256_s5():
    return rlh.generateGaussPSF((256,256,256), 5)

@pytest.fixture
def psf_gauss_w32_s5():
    return rlh.generateGaussPSF((32,32,32), 5)

@pytest.fixture
def psf_gauss_w16_s3():
    return rlh.generateGaussPSF((16,16,16), 3)

@pytest.fixture
def data_with_cubes_256(psf_gauss_w16_s3):
    #Add a few cubes in grid-like locations
    datashape = (256,256,256)
    cubesize=2
    cubespacing=16
    amplitude = 1e6

    print(f"Creating 3D data with shape {datashape}.")

    data = np.zeros(datashape, dtype=np.float32)
    for iz in range(int(cubespacing/2) ,datashape[0],cubespacing):
        for iy in range(int(cubespacing/2) ,datashape[1],cubespacing):
            for ix in range(int(cubespacing/2) ,datashape[2],cubespacing):
                data[iz:iz+cubesize , iy:iy+cubesize , ix:ix+cubesize] = np.ones((cubesize,cubesize,cubesize))*amplitude
    
    psf = psf_gauss_w16_s3

    print(f"psf.shape: {psf.shape}")
    print(f"psf max, min: {psf.max()},{psf.min()}")

    #convolve
    print("Convolving with PSF.")
    data_convolved = scipy.signal.convolve(data, psf, mode='same')
    #Fix zeros
    data_convolved = np.where(data_convolved<0, 0, data_convolved)
    print(f"data_convolved max, min: {data_convolved.max()},{data_convolved.min()}")

    print("Adding poisson noise to data.")
    rng = np.random.default_rng()
    data_convolved_noised = rng.poisson(lam = data_convolved)

    return(data_convolved_noised)