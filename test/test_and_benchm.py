'''
Benchmark RedLionfish deconvolution on a large 3D data.
Generate data here and then run the deconvolution.
Prints the time it takes to complete the deconvolution
'''

import time
import numpy as np
import math
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
rlfocl.printGPUInfo()

#Run a deconvolution

#Create a small 2d array to do FFT
datashape = (256,256,256)
print(f"Creating 3D data with shape {datashape}.")
data = np.zeros(datashape, dtype=np.float32)

#Add a few cubes in grid-like locations
cubesize=2
cubespacing=16
for iz in range(int(cubespacing/2) ,datashape[0],cubespacing):
    for iy in range(int(cubespacing/2) ,datashape[1],cubespacing):
        for ix in range(int(cubespacing/2) ,datashape[2],cubespacing):
            data[iz:iz+cubesize , iy:iy+cubesize , ix:ix+cubesize] = np.ones((cubesize,cubesize,cubesize))

print("Generating PSF data.")

def generateGaussPSF():
    shape = (32,32,32)
    z_mg = np.linspace(-1,1,shape[0], dtype=np.float32)
    y_mg = np.linspace(-1,1,shape[0], dtype=np.float32)
    x_mg = np.linspace(-1,1,shape[0], dtype=np.float32)

    z_mg, y_mg, x_mg = np.meshgrid( z_mg, y_mg, x_mg)

    sigma = 0.2

    data = 1/sigma/math.sqrt(2*math.pi) * np.exp( -0.5 * ( np.square(z_mg) + np.square(y_mg) + np.square(x_mg) ) / sigma/sigma )

    return data

datapsf = generateGaussPSF()

print("Convoluting, using scipy.signal")
import scipy.signal
data_convolved = scipy.signal.convolve(data, datapsf, mode='same')
#Normalises it to range 0-255
data_convolved = (data_convolved - data_convolved.min()) / (data_convolved.max() - data_convolved.min())*255

print("Adding poisson noise to data.")
rng = np.random.default_rng()
data_convolved_noised = rng.poisson(lam = data_convolved)

data_convolved_noised_uint8 = ((data_convolved_noised - data_convolved_noised.min()) / ( data_convolved_noised.max() - data_convolved_noised.min() ) *255 ).astype(np.uint8)

niter = 5

print(f"RL deconvolution, niter = {niter}")

import time
print("CPU")

t0 = time.time()
res_CPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='cpu')
t1=time.time()

print(f"RL deconvolution using CPU took {t1-t0} s")

print("GPU")
t0 = time.time()
res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='gpu')
t1=time.time()
print(f"RL deconvolution using GPU took {t1-t0} s")

print()


#With larger data

#Create a small 2d array to do FFT
datashape = (1024,1024,64)
print(f"Creating large 3D data with shape {datashape}.")
data = np.zeros(datashape, dtype=np.float32)

#Add a few cubes in grid-like locations
cubesize=2
cubespacing=16
for iz in range(int(cubespacing/2) ,datashape[0],cubespacing):
    for iy in range(int(cubespacing/2) ,datashape[1],cubespacing):
        for ix in range(int(cubespacing/2) ,datashape[2],cubespacing):
            data[iz:iz+cubesize , iy:iy+cubesize , ix:ix+cubesize] = np.ones((cubesize,cubesize,cubesize))

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
res_CPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='cpu')
t1=time.time()

print(f"RL deconvolution using CPU took {t1-t0} s")

print("GPU")
t0 = time.time()
res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=10, method='gpu')
t1=time.time()
print(f"RL deconvolution using GPU took {t1-t0} s")
