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

#Entry point for doing RL 3D deconvolution using either CPU or GPU

#TODO: A lot of these parameters can be passed as **kwargs.
#TODO: Test GPU acceleration on MacOS
#TODO: Consider open files using scipy.imread rather than the tifffile modules


#from RedLionfishDeconv import helperfunctions
#from helperfunctions import *
#import helperfunctions
import logging
import numpy as np

def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False):
    '''
        Richardson-Lucy deconvolution of 3D data.
        It does NOT use the skimage.image.restoration.rishardson_lucy.
        Iteration uses FFT-based convolutions, either using CPU (scipy) or GPU (Reikna OpenCL)
        
        Parameters:
            data_np: 3d volume of data as numpy array
            psf_np: point-spread-function to use for deconvolution
            niter: number of iterations to perform
            method: 'gpu' to use Reikna OpenCL , 'cpu' to use Scipy. 
            useBlockAlgorithm: 'gpu' only, forces to use block algorithm. Result may show boundary effects.
            callbkTickFunc: function to use to provide tick update during the RL algorithm. This can be either each iteration step or in case of block algorithm, each block calculation
            resAsUint8: Set to return the result of the RL deconvolution as a np.uint8 array. Useful for displaying in napari for example. Note that it will adjust minimum and maximum to ful range 0-255.
                Set to False to return result as default np.float32 format.
    '''
    from RedLionfishDeconv import helperfunctions

    logging.info(f"doRLDeconvolutionFromNpArrays(), niter={niter} , method={method} , useBlockAlgorithm={useBlockAlgorithm}, resAsUint8={resAsUint8}")

    if data_np.ndim !=3:
        logging.info("Data is not 3 dimensional. Exiting.")
        return None
    if psf_np.ndim != 3:
        logging.info("PSF is not 3-dimensional. Exiting.")
        return None

    resRL = None

    if method=='gpu':
        import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlreikna
        if rlreikna.isReiknaAvailable:
            logging.info("Reikna package is available")
            #Great, use the reikna then
            if not useBlockAlgorithm:
                #Use standard RL OCL version
                logging.info("Trying the OCL non-block RL deconvolution.")
                try:
                    resRL = rlreikna.nonBlock_RLDeconvolutionReiknaOCL(data_np, psf_np, niter=niter, callbkTickFunc=callbkTickFunc)
                except Exception as e:
                    #Probably out of memory error, fallback to block algorithm
                    logging.info("nonBlock_RLDeconvolutionReiknaOCL() failed (GPU).")
                    logging.info(e)
                    useBlockAlgorithm= True
            if useBlockAlgorithm:
                logging.info("Trying the OCL block RL deconvolution algorithm.")
                bKeepTrying=True
                blocksize=512

                #Check psf size is not too large
                if np.max(np.array(psf_np.shape))< int(blocksize/2):
                    while bKeepTrying:
                        try:
                            resRL = rlreikna.block_RLDeconv3DReiknaOCL4(data_np , psf_np,niter=niter,max_dim_size=blocksize, callbkTickFunc=callbkTickFunc)
                            bKeepTrying=False
                        except Exception as e:
                            #Error doing previous calculation, reduce block size
                            logging.info(f"Error: block_RLDeconv3DReiknaOCL4 with blocksize={blocksize} failed (GPU). Will try to halve blocksize.")
                            logging.info(e)
                            if blocksize>=128 :
                                blocksize = blocksize//2
                                bKeepTrying=True
                            else:
                                #No point reducing the block size to smaller, fall back to CPU
                                bKeepTrying=False
                                method = 'cpu'
                                logging.info('GPU calculation failed, falling back to CPU.')
                else:
                    logging.info('PSF shape is too large for doing block iteration. Falling to CPU.')
                    method = 'cpu'
                    

        else:
            logging.info("Reikna is not available, falling back to CPU scipy calculation")
            method = 'cpu'
        
    if method == 'cpu':
        from . import RLDeconv3DScipy as rlcpu
        try:
            resRL = rlcpu.doRLDeconvolution_DL2_4(data_np, psf_np, niter=niter, callbkTickFunc=callbkTickFunc)
        except Exception as e:
            logging.info("doRLDeconvolution_DL2_4 failed (CPU) with error:")
            logging.info(str(e))
    
    #TODO: use dask

    if resAsUint8:
        resRL = helperfunctions.convertToUint8AndFullRange(resRL)

    return resRL

def doRLDeconvolutionFromFiles(datapath, psfpath, niter, savepath=None):
    '''
    Opens tiff files and runs Richardson-Lucy deconvoultion
    savepath: location and filename of the tiff file that will store the result

    Returns:
        res_np: result of the RL deconvolution
    '''
    import tifffile as tf

    #Check al info is ok
    data_np = np.array(tf.imread(datapath))
    psf_np = np.array(tf.imread(psfpath))

    res_np = doRLDeconvolutionFromNpArrays(data_np, psf_np, niter=niter, resAsUint8=True)

    if (not res_np is None):
        logging.info("res_np collected")
        if (not savepath is None):
            logging.info(f"Saving data to {savepath}")
            tf.imsave(savepath, res_np)
    
    return res_np

