#Entry point for doing RL 3D deconvolution using either CPU or GPU

from RedLionfishDeconv import helperfunctions


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
            resAsUint8: Set to to return the result of the deconvolution as a np.uint8 array. Useful for displaying in napari for example. Note that it will adjust minimum and maximum to ful range 0-255.
                Set to False to return result as deault np.float32 format.
    '''
    resRL = None

    if method=='gpu':
        import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlreikna
        if rlreikna.isReiknaAvailable:
            #Great, use the reikna then
            if not useBlockAlgorithm:
                #Use standard RL OCL version
                try:
                    resRL = rlreikna.nonBlock_RLDeconvolutionReiknaOCL(data_np, psf_np, niter=niter, callbkTickFunc=callbkTickFunc)
                except Exception as e:
                    #Probably out of memory error, fallback to block algorithm
                    print("Error: nonBlock_RLDeconvolutionReiknaOCL failed (GPU). Will try next to use block deconvolution.")
                    print(e)
                    useBlockAlgorithm= True
            if useBlockAlgorithm:
                bKeepTrying=True
                blocksize=512
                while bKeepTrying:
                    try:
                        resRL = rlreikna.block_RLDeconv3DReiknaOCL4(data_np , psf_np,niter=niter,max_dim_size=blocksize, callbkTickFunc=callbkTickFunc)
                        bKeepTrying=False
                    except Exception as e:
                        #Error doing previous calculation, reduce block size
                        print(f"Error: block_RLDeconv3DReiknaOCL4 with blocksize={blocksize} failed (GPU). Will try to halve blocksize.")
                        print(e)
                        if blocksize>=128 :
                            blocksize = blocksize//2
                            bKeepTrying=True
                        else:
                            #No point reducing the block size to smaller, fall back to CPU
                            bKeepTrying=False
                            method = 'cpu'
                            print('GPU calculation failed, falling back to CPU.')

        else:
            print ("Reikna is not available, falling back to CPU scipy calculation")
            method = 'cpu'
        
    if method == 'cpu':
        from . import RLDeconv3DScipy as rlcpu
        try:
            resRL = rlcpu.doRLDeconvolution_DL2_4(data_np, psf_np, niter=niter, callbkTickFunc=callbkTickFunc)
        except Exception as e:
            print("doRLDeconvolution_DL2_4 failed (CPU) with error:")
            print(str(e))

    if resAsUint8:
        resRL = helperfunctions.convertToUint8AndFullRange(resRL)

    return resRL

#TODO: Ability to run from command line providing tiff (or other) filenames
