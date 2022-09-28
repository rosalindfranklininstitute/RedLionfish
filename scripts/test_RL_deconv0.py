

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
import RedLionfishDeconv.helperfunctions
rlfocl.printGPUInfo()

datashape = (834, 300, 2048) #Pradeep Rajasekhar issue
datapsfshape = (93, 205, 205)

print(f"Creating 3D data with shape {datashape}.")
data = np.zeros(datashape, dtype=np.float32)
#Add a few cubes in grid-like locations
cubesize=2
cubespacing=16
for iz in range(int(cubespacing/2) ,datashape[0],cubespacing):
    for iy in range(int(cubespacing/2) ,datashape[1],cubespacing):
        for ix in range(int(cubespacing/2) ,datashape[2],cubespacing):
            data[iz:iz+cubesize , iy:iy+cubesize , ix:ix+cubesize] = np.ones((cubesize,cubesize,cubesize))

psf = RedLionfishDeconv.helperfunctions.generateGaussPSF( datapsfshape )


print(f"data.shape: {data.shape}")
print(f"psf.shape: {psf.shape}")

#test

res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=3)


print(f"res_gpu.shape: {res_gpu.shape}")

