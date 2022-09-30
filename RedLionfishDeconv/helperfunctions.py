'''
Copyright 2021 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import logging
import math

#Helper function
def change3DSizeTo(data, shape):
    '''
    Resizes data to the new shape keeping data centralized.
    If data size is smaller, then egdes are cropped.
    If data is larger, data will be padded with zeros.
    '''
    #based in size() code found in
    #https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/e9af0aba493ba137d70877154648e5583e376a81/src/main/java/signal/RealSignal.java#L531

    logging.info(f"change3DSizeTo() , shape:{shape}")
    
    mz,my, mx = shape
    nz,ny,nx = data.shape

    psf1 = np.zeros( shape , dtype=data.dtype)
    
    vx = min(nx,mx)
    vy = min(ny,my)
    vz = min(nz, mz)
    ox = int((mx - nx) / 2)
    oy = int((my - ny) / 2)
    oz = int((mz - nz) / 2)

    (pz0, qz0) = (oz, 0) if oz>=0 else (0, -oz)
    (py0, qy0) = (oy, 0) if oy>=0 else (0, -oy)
    (px0, qx0) = (ox, 0) if ox>=0 else (0, -ox)

    psf1[ pz0:pz0+vz , py0:py0+vy , px0:px0+vx ] = data[ qz0:qz0+vz , qy0:qy0+vy ,  qx0:qx0+vx]

    return psf1

def change3DSizeTo_KeepOrigin(data, shape):
    '''
    Resizes data to the new shape keeping data near origin.
    If data size is smaller, then egdes are cropped.
    If data is larger, data will be padded with zeros.
    '''

    logging.info(f"change3DSizeKeepOriginTo() , shape:{shape}")
    
    mz,my, mx = shape
    nz,ny,nx = data.shape

    psf1 = np.zeros( shape , dtype=data.dtype)
    
    vx = min(nx,mx)
    vy = min(ny,my)
    vz = min(nz, mz)

    psf1[0:vz , 0:vy ,  0:vx] = data[ 0:vz , 0:vy ,  0:vx]

    return psf1

#An implementation of the circular() routine from DeconvolutionLab2 that shifts the data centre of the psf
#to the corners of the volume.
#This is important so that convolution of data with psf is will not shift features (like beads).
#It is assumeed that the PSF data has the psf-origin at the center of the data 
def circulify3D(data3D):
    #Check data is 3D
    if data3D.ndim != 3:
        raise Exception("Error, input data is not 3-dimensional")

    data0 = np.array(data3D)

    shape = data0.shape

    zdimhalf0 = int(shape[0]/2)
    zdimhalf1 = shape[0]-zdimhalf0

    ydimhalf0 = int(shape[1]/2)
    ydimhalf1 = shape[1]-ydimhalf0

    xdimhalf0 = int(shape[2]/2)
    xdimhalf1 = shape[2]-xdimhalf0

    if zdimhalf0 >=1 and ydimhalf0>=1 and xdimhalf0>=1:
        data0[zdimhalf1: , ydimhalf1:, xdimhalf1:] = data3D[:zdimhalf0 , :ydimhalf0 , :xdimhalf0 ] #cube 1
        data0[zdimhalf1: , ydimhalf1:, :xdimhalf1] = data3D[:zdimhalf0 , :ydimhalf0 , xdimhalf0: ] #cube 2
        data0[zdimhalf1: , :ydimhalf1, :xdimhalf1] = data3D[:zdimhalf0 , ydimhalf0: , xdimhalf0: ] #cube 3
        data0[zdimhalf1: , :ydimhalf1, xdimhalf1:] = data3D[:zdimhalf0 , ydimhalf0: , :xdimhalf0 ] #cube 4
        data0[:zdimhalf1 , ydimhalf1:, xdimhalf1:] = data3D[zdimhalf0: , :ydimhalf0 , :xdimhalf0 ] #cube 5
        data0[:zdimhalf1 , ydimhalf1:, :xdimhalf1] = data3D[zdimhalf0: , :ydimhalf0 , xdimhalf0: ] #cube 6
        data0[:zdimhalf1 , :ydimhalf1, :xdimhalf1] = data3D[zdimhalf0: , ydimhalf0: , xdimhalf0: ] #cube 7
        data0[:zdimhalf1 , :ydimhalf1, xdimhalf1:] = data3D[zdimhalf0: , ydimhalf0: , :xdimhalf0 ] #cube 8
    
    return data0

def convertToFloat32AndNormalise(data, normaliseType=None, bResetZero=False):
    '''
    Converts data to float32 format and optionally resets zero and normalises
    Parameters:
        data
        normalisetype: None for no normalisation, 'max' normalises to max, 'sum' to normalise to sum of whole volume
        bResetZero: flags to whether shift the minimum value to zero
    '''
    if data is None:
        return None
    
    ret= data.astype(np.float32)

    vmin= 0
    if bResetZero:
        vmin = ret.min()
    ret = ret-vmin

    if not normaliseType is None:
        normcorr = 1.0
        if normaliseType == 'max':
            #Normalise to maximum value
            normcorr= ret.max()
        else:
            #normalise to sum
            normcorr = np.sum(ret)
        #Check for zero before division
        if normcorr==0.0:
            logging.info("normcorr = 0. Normalisation will be skipped to prevent division by zero.")
        else:
            ret = ret/normcorr
    return ret

def convertToUint8AndFullRange(data):
    if data is None:
        return None
        
    res_256 = convertToFloat32AndNormalise(data, normaliseType='max', bResetZero=True)*255
    res_uint8 = res_256.astype(np.uint8)
    return res_uint8

def generateGaussPSF(shape, sigma=10):
    #print(shape)
    
    shape_mid = np.floor(np.array(shape)/2)
    z = np.arange(-shape_mid[0], shape_mid[0],step=1, dtype=np.float32 )
    y = np.arange(-shape_mid[1], shape_mid[1],step=1, dtype=np.float32 )
    x = np.arange(-shape_mid[2], shape_mid[2],step=1, dtype=np.float32 )

    z_mg, y_mg, x_mg = np.meshgrid( z, y, x)

    data = 1/sigma/math.sqrt(2*math.pi) * np.exp( -0.5 * ( np.square(z_mg) + np.square(y_mg) + np.square(x_mg) ) / sigma/sigma )

    return data

