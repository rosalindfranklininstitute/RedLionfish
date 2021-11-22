#Several versions of Richardson-Lucy deconvolution using scipy.signal convolution

import numpy as np
import scipy.fft

from .helperfunctions import *
import logging

def doRLDeconvolution12(data_np , psf_np , *, niter=10, callbkTickFunc=None):
    #RL deconvolution based on doRLDeconvolution11()
    # Reverse engineered parts of https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.richardson_lucy
    # and also scipy.signal convolution fft code for faster processing
    #https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/signaltools.py#L1293-L1413
    #for mode='same', method='fft', fftconvolution() #
    #There is a small difference here compared to scikit-image version in the sense that the psf is normalised to the sum.
    #This version tries to do all the work with arrays with preferred size of next_fast_len
    #Only at the end of the calculation it gets the shape needed

    #Convert and normalise
    data_np_norm = convertToFloat32AndNormalise(data_np)
    psf_np_norm = convertToFloat32AndNormalise(psf_np , normaliseType='sum', bResetZero=False) #Normalise to sum

    s1 = data_np_norm.shape
    s2 = psf_np_norm.shape

    shape = [(s1[i] + s2[i] - 1) for i in range(data_np_norm.ndim)]
    fshape = [scipy.fft.next_fast_len(shape[a], True) for a in range(len(shape))]

    #Resize both the data and psf to the preferred fshape
    data_shaped = change3DSizeTo(data_np_norm, fshape)
    psf_shaped = change3DSizeTo(psf_np_norm, fshape)

    #Circulify psf
    psf1 = circulify3D(psf_shaped)

    #Precalculated psf_fft
    psf_fft= scipy.fft.rfftn(psf1)
    psf_flip = np.flip(psf1)
    psf_flip_fft = scipy.fft.rfftn(psf_flip) #FFT of the flipped psf

    xn1 = np.array(data_shaped) #initialize copy
    for i in range(niter):
        xn=xn1
        xn_fft = scipy.fft.rfftn(xn) 
        U0 = np.multiply(xn_fft ,psf_fft)
        u0 = scipy.fft.irfftn(U0)

        u0 = np.where(u0==0, 1e-14, u0) #Fix zero values giving a finite value
            
        p = np.divide(data_shaped,u0)

        U1 = scipy.fft.rfftn(p)
        U2 = np.multiply(U1 , psf_flip_fft)
        u2 = scipy.fft.irfftn(U2)

        xn1 = np.multiply(xn, u2)

        if not callbkTickFunc is None:
            callbkTickFunc()
    
    #Get the central part of the result
    #xn1_central = _centered(xn1, s1)
    xn1_central = change3DSizeTo(xn1, s1) #To original shape

    # data_deconv_norm_256 = convertAndNormalise(xn1_central)*256
    # data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
    
    data_deconv_uint8 = convertToUint8AndFullRange(xn1_central)
    return data_deconv_uint8


#default
def doRLDeconvolution_DL2_4(data_np , psf_np ,*, niter=10, callbkTickFunc=None):
    #RL deconvolution based in DeconvolutionLab2 with optional parameter for normalising inputs
    #Mimics DeconvolutionLab2 (DL2) as best as possible
    #https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    #However it prepares the psf to have the same size as the input data
    #In DeconvolutionLab2 only psf is normalised to psf sum
    #
    #This version uses scipy
    #It is modified from the DeconvolutionLab2 algorithm
    # the fft of the psf is not conjugated, but instead is flipped before conjugation
    # The developers probably thought that the 'reality condition' applies
    # FT[f](-y) = FT[f]*
    # but that is not true, as code tests proved (tests.ipynb)
    # In this version, as in the original formula for RL, the flipped psf is used for the second convolution
    # (see tests.ipynb)
    
    logging.info("doRLDeconvolution_DL2_4() (CPU)")
    #Convert and normalise
    data_np_norm =convertToFloat32AndNormalise(data_np) #don't normalise

    #Check last axis is even size, otherwise it will give error
    if data_np_norm.shape[-1] %2 != 0: #odd number
        logging.info("Data last axis size is an odd number. Padding with zeros to make size even to prevent errors")
        s = data_np_norm.shape
        shape_fix = (s[0], s[1], s[2]+1)
        data_fix = np.empty(shape_fix, dtype = data_np_norm.dtype)
        data_fix[:,:,:-1] = data_np_norm[:,:,:]
        data_fix[:,:,-1] = 0
        data_np_norm = data_fix

    # psf_norm = psf0/sum#
    psf_norm = convertToFloat32AndNormalise(psf_np, normaliseType='sum',bResetZero=False)

    psf0 = change3DSizeTo(psf_norm, data_np_norm.shape)
    #psf1 should have the same dimensions as data

    #Do circulirisation of psf data before starting the RL algorithm
    psf1 = circulify3D(psf0)

    #Precalculated psf_fft
    psf_fft= scipy.fft.rfftn(psf1)
    psf_flip = np.flip(psf1)
    psf_flip_fft = scipy.fft.rfftn(psf_flip) #FFT of the flipped psf

    xn1 = np.array(data_np_norm) #initialize copy
    for i in range(niter):
        xn=xn1
        xn_fft = scipy.fft.rfftn(xn)  
        U0 = np.multiply(xn_fft ,psf_fft)
        u0 = scipy.fft.irfftn(U0)

        u0 = np.where(u0==0, 1e-14, u0) #Fix zero values giving a finite value

        p = np.divide(data_np_norm,u0)

        U1 = scipy.fft.rfftn(p)
        U2 = np.multiply(U1 , psf_flip_fft)
        u2 = scipy.fft.irfftn(U2)

        xn1 = np.multiply(xn, u2)

        if not callbkTickFunc is None:
            callbkTickFunc()

    data_deconv_uint8 = convertToUint8AndFullRange(xn1)
    return data_deconv_uint8