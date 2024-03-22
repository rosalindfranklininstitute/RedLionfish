"""
Benchmark RedLionfish deconvolution on a large 3D data.
Generate data here and then run the deconvolution.
Prints the time it takes to complete the deconvolution
"""

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

import pandas as pd

df_benchm_results = pd.DataFrame(
    {
        "shape": [],
        "method": [],
        "GPU/CPU": [],
        "blocking": [],
        'iterations':[],
        "time / s": [],
    }
)

print(len(df_benchm_results))

#Get OpenCL information
print("Checking RLDeconv3DReiknaOCL")
import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlfocl
import RedLionfishDeconv.helperfunctions
rlfocl.printGPUInfo()

testWithGPU = False
if rlfocl.isReiknaAvailable:
    testWithGPU = True
    print("Reikna available. Will run GPU tests.")
else:
    print("Reikna not available. GPU deconvolution tests will be skipped.")
    
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

datapsf = RedLionfishDeconv.helperfunctions.generateGaussPSF( (32,32,32) )

print("Convoluting, using scipy.signal")
import scipy.signal
data_convolved = scipy.signal.convolve(data, datapsf, mode='same')
#Normalises it to range 0-255
data_convolved = (data_convolved - data_convolved.min()) / (data_convolved.max() - data_convolved.min())*255

print("Adding poisson noise to data.")
rng = np.random.default_rng()
data_convolved_noised = rng.poisson(lam = data_convolved)

data_convolved_noised_uint8 = ((data_convolved_noised - data_convolved_noised.min()) / ( data_convolved_noised.max() - data_convolved_noised.min() ) *255 ).astype(np.uint8)

niter = 10

print(f"RL deconvolution, niter = {niter}")

import time
print("CPU")

t0 = time.time()
res_CPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='cpu')
t1=time.time()

deltat_s = t1-t0
print(f"RL deconvolution using CPU took {deltat_s} s")

# Add new row
df_benchm_results.loc[len(df_benchm_results)]={
    "shape": str(datashape),
    "method": "doRLDeconvolutionFromNpArrays",
    "GPU/CPU": "CPU",
    "blocking": "?",
    'iterations':niter,
    "time / s": deltat_s,
}



if testWithGPU:
    print()
    print("GPU")
    t0 = time.time()
    res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='gpu')
    t1=time.time()
    
    deltat_s = t1-t0
    print(f"RL deconvolution using GPU took {deltat_s} s")

    df_benchm_results.loc[len(df_benchm_results)]={
        "shape": str(datashape),
        "method": "doRLDeconvolutionFromNpArrays - OpenCL CUDA",
        "GPU/CPU": "GPU",
        "blocking": "?",
        'iterations':niter,
        "time / s": deltat_s,
    }


    import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlocl

    #Use OpenCL CPU
    print("OpenCL CPU")
    t0 = time.time()
    rlocl.RL_CL_PLATFORM_PREFERENCE="intel" # sets to use intel CPU as openCL device
    res_ocl_intel = rlocl.nonBlock_RLDeconvolutionReiknaOCL(data,datapsf, niter=niter)
    #res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='gpu')
    t1=time.time()
    deltat_s = t1-t0
    print(f"RL deconvolution using OpenCL CPU took {deltat_s} s")

    df_benchm_results.loc[len(df_benchm_results)]={
        "shape": str(datashape),
        "method": "nonBlock_RLDeconvolutionReiknaOCL - OpenCL CPU",
        "GPU/CPU": "CPU",
        "blocking": "No",
        'iterations':niter,
        "time / s": deltat_s,
    }


import skimage.restoration
print("RL calculation using skimage.restoration.richardson_lucy()")
t0 = time.time()
res_skimage = skimage.restoration.richardson_lucy(data,datapsf, num_iter = niter, clip=False)
t1=time.time()
deltat_s = t1-t0
print(f"RL deconvolution using skimage.restoration.richardson_lucy() took {deltat_s} s")
# Add new row
df_benchm_results.loc[len(df_benchm_results)]={
    "shape": str(datashape),
    "method": "skimage.restoration.richardson_lucy()",
    "GPU/CPU": "CPU",
    "blocking": "No",
    'iterations':niter,
    "time / s": deltat_s,
}



print()



#With larger data

#Create a larger 3 array to do FFT
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
res_CPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='cpu')
t1=time.time()

deltat_s = t1-t0
print(f"RL deconvolution using CPU took {deltat_s} s")

df_benchm_results.loc[len(df_benchm_results)]={
    "shape": str(datashape),
    "method": "doRLDeconvolutionFromNpArrays - CPU",
    "GPU/CPU": "CPU",
    "blocking": "?",
    'iterations':niter,
    "time / s": deltat_s,
}



if testWithGPU:
    print()
    print("GPU")
    t0 = time.time()
    rlocl.RL_CL_PLATFORM_PREFERENCE="cuda" # ensure it uses CUDA
    res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='gpu')
    t1=time.time()
    
    deltat_s = t1-t0
    print(f"RL deconvolution using GPU took {deltat_s} s")

    df_benchm_results.loc[len(df_benchm_results)]={
        "shape": str(datashape),
        "method": "doRLDeconvolutionFromNpArrays - OpenCL CUDA",
        "GPU/CPU": "GPU",
        "blocking": "?",
        'iterations':niter,
        "time / s": deltat_s,
    }


    import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlocl

    #Use OpenCL CPU
    print("OpenCL CPU")
    t0 = time.time()
    rlocl.RL_CL_PLATFORM_PREFERENCE="intel"
    #res_ocl_intel = rlocl.nonBlock_RLDeconvolutionReiknaOCL(data,datapsf, niter=niter)
    res_ocl_intel = rlocl.block_RLDeconv3DReiknaOCL(data,datapsf, niter=niter)
    #res_GPU = rl.RLDeconvolve.doRLDeconvolutionFromNpArrays(data_convolved_noised_uint8, datapsf, niter=niter, method='gpu')
    t1=time.time()
    deltat_s = t1-t0
    print(f"RL deconvolution using OpenCL CPU took {deltat_s} s")

    df_benchm_results.loc[len(df_benchm_results)]={
        "shape": str(datashape),
        "method": "block_RLDeconv3DReiknaOCL - OpenCL CPU",
        "GPU/CPU": "CPU",
        "blocking": "yes",
        'iterations':niter,
        "time / s": deltat_s,
    }

print("RL calculation using skimage.restoration.richardson_lucy()")
t0 = time.time()
res_skimage = skimage.restoration.richardson_lucy(data,datapsf, num_iter = niter, clip=False)
t1=time.time()
deltat_s = t1-t0
print(f"RL deconvolution using skimage.restoration.richardson_lucy() took {deltat_s} s")
# Add new row
df_benchm_results.loc[len(df_benchm_results)]={
    "shape": str(datashape),
    "method": "skimage.restoration.richardson_lucy()",
    "GPU/CPU": "CPU",
    "blocking": "No",
    'iterations':niter,
    "time / s": deltat_s,
}

print(df_benchm_results)

print()

print(df_benchm_results.to_csv())