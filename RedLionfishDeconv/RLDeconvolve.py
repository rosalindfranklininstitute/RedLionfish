#Entry point for doing RL 3D deconvolution using either CPU or GPU

#TODO: A lot of these parameters can be passed as **kwargs.
#TODO: Test GPU acceleration on MacOS
#TODO: Convert print() to logging.info
#TODO: Consider open files using scipy.imread rather than the tifffile modules

from numpy.lib.npyio import save
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
            resAsUint8: Set to return the result of the RL deconvolution as a np.uint8 array. Useful for displaying in napari for example. Note that it will adjust minimum and maximum to ful range 0-255.
                Set to False to return result as default np.float32 format.
    '''

    print(f"doRLDeconvolutionFromNpArrays(), niter={niter} , method={method} , useBlockAlgorithm={useBlockAlgorithm}, resAsUint8={resAsUint8}")

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

def doRLDeconvolutionFromFiles(datapath, psfpath, niter, savepath=None):
    import tifffile as tf
    import sys
    import numpy as np

    #Check al info is ok
    data_np = np.array(tf.imread(datapath))
    if data_np.ndim !=3:
        print("Data is not 3 dimensional. Exiting.")
        sys.exit

    psf_np = np.array(tf.imread(psfpath))
    if psf_np.ndim != 3:
        print("Psf is not 3-dimensional. Exiting.")
        sys.exit()

    res_np = doRLDeconvolutionFromNpArrays(data_np, psf_np, niter=niter, resAsUint8=True)

    print("res_np collected")

    if (not res_np is None) and (not savepath is None):
        print(f"Saving data to {savepath}")
        tf.imsave(savepath, res_np)
    
    return res_np



# Ability to run from command line providing tiff (or other) filenames
def main():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Richardson-Lucy deconvolution of 3D data.")
    parser.add_argument("data3Dpath", help="Input 3d data file, tiff format")
    parser.add_argument("psf3Dpath" , help="Input 3d psf/otf file, tiff format")
    parser.add_argument("iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--outpath" , "-o", help="output filanme")
    parser.add_argument("--method" , help = "force gpu or cpu method. (not implemented yet, automatic, trying gpu first).")

    args= parser.parse_args()

    data3Dpath = args.data3Dpath
    #check files data3Dpath and psf3Dpath exist
    if not os.path.exists(data3Dpath) :
        print(f"File {data3Dpath} could not be found. Exiting.")
        sys.exit()

    psf3Dpath = args.psf3Dpath
    if not os.path.exists(psf3Dpath) :
        print(f"File {psf3Dpath} could not be found. Exiting.")
        sys.exit()

    iterations = args.iterations
    if iterations <=0:
        print(f"Invalid number of iterations {iterations}")
        sys.exit
    
    #setup the filename.
    # if not provided use the data filname with added _it<iterations>.tiff
    outpath = args.outpath
    if outpath is None:
        pathhead, pathtail = os.path.split(args.data3Dpath)
        pathname , ext = os.path.splitext(pathtail)
        outpath = pathname + "_it" + str(iterations) + ".tiff"

    doRLDeconvolutionFromFiles(data3Dpath, psf3Dpath, iterations, savepath=outpath)


if __name__ == "__main__":
    # Run if called from the command line
    main()
