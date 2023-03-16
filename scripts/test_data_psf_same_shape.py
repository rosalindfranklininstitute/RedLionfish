"""
Benchmark RedLionfish deconvolution on a large 3D data.
Generate data here and then run the deconvolution.
Prints the time it takes to complete the deconvolution
"""

import time
import numpy as np
import scipy.signal

import logging
logging.basicConfig(level=logging.INFO)
#logging.info('test logging info')

import RedLionfishDeconv as rl
#If there is an error with the package it will throw an error here
print("import RedLionfishDeconv succeeded")

#Get OpenCL information
print("Checking RLDeconv3DReiknaOCL")
import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlfocl
import RedLionfishDeconv.helperfunctions
rlfocl.printGPUInfo()

niter = 5

print("Testing weird shape and data.shape = psf.shape")
sameshape = (48,57,93)

print(f"Generating data with shape ={sameshape}")

data = np.zeros(sameshape, dtype=np.float32)
#Add a few cubes in grid-like locations
cubesize=2
cubespacing=16
for iz in range(int(cubespacing/2) ,sameshape[0],cubespacing):
    for iy in range(int(cubespacing/2) ,sameshape[1],cubespacing):
        for ix in range(int(cubespacing/2) ,sameshape[2],cubespacing):
            try:
                data[iz:iz+cubesize , iy:iy+cubesize , ix:ix+cubesize] = np.ones((cubesize,cubesize,cubesize))
            except Exception as e:
                pass #do not put the small cube in case of error

print(f"Generating psf with shape {sameshape}")

datapsf = RedLionfishDeconv.helperfunctions.generateGaussPSF( sameshape )

print("Convoluting 3D data with the psf, using scipy.signal")
data_convolved = scipy.signal.convolve(data, datapsf, mode='same')
#Normalises it to range 0-255
data_convolved = (data_convolved - data_convolved.min()) / (data_convolved.max() - data_convolved.min())*255

print("Adding poisson noise to data.")
rng = np.random.default_rng()
data_convolved_noised = rng.poisson(lam = data_convolved)

data_convolved_noised_uint8 = ((data_convolved_noised - data_convolved_noised.min()) / ( data_convolved_noised.max() - data_convolved_noised.min() ) *255 ).astype(np.uint8)

print(f"RL deconvolution, niter = {niter}")
print("CPU")
t0 = time.time()
res_CPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='cpu', resAsUint8=True)
t1=time.time()
print(f"RL deconvolution using CPU took {t1-t0} s")

print("GPU")
t0 = time.time()
res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='gpu', resAsUint8=True)
t1=time.time()
print(f"RL deconvolution using GPU took {t1-t0} s")

datapsf_uint8 = RedLionfishDeconv.helperfunctions.convertToUint8AndFullRange(datapsf)

print("skimage.restoration.richardson_lucy")
#Compare with skimage
import skimage.restoration
t0 = time.time()
res_skimage = skimage.restoration.richardson_lucy(data, datapsf, iterations=10,clip=False)
t1=time.time()
print(f"RL deconvolution using scikitimage took {t1-t0} s")

import napari
nv = napari.view_image(res_GPU, ndisplay=3)
nv.add_image(res_CPU)
nv.add_image(data_convolved_noised_uint8)

nv.add_image(datapsf_uint8)

nv.add_image(res_skimage)

napari.run()
