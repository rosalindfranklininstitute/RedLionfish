#Richardson-Lucy using Reikna OpenCL

#Usage will be through a class, whereby user needs to set the shape for the 3D calculation volume
#And then set the psf
#And then run

import logging
#This could be useful to check all ok before trying to run GPU accelerated code
# If not, user can fallback to CPU version using scipy
isReiknaAvailable = False
try:
    import reikna.cluda as cluda
    from reikna.fft import FFT
    from reikna.cluda import functions, dtypes

    apitest = cluda.ocl_api() #Just checking it can get an opencl device (may not be the fastest available!)
    #If it fails to get a device, it will throw an error
    isReiknaAvailable=True
    del(apitest)

except Exception as e:
    logging.error("Failed to setup Reikna with OpenCL.")
    logging.error(e)
    isReiknaAvailable=False

import logging
import numpy as np
from .helperfunctions import *

class RLDeconv3DReiknaOCL:
    '''
    Class that helps setting up and run Richardson Lucy deconvolution using Reikna opencl FFT
    To use this, construct an instance by providing the shape information of the data that you want to do the running the RL algorithm.
    This will setup the calculations in the device, which include compilation steps.
    Before running the RL, user has to set the PSF using function setPSF().
    This will setup the psf internally for making calculations faster,
    such as resizing psf to datahshape, circulify, and precalculate ffts for convolutions.

    Run the RL algorithm by calling doRLDeconvolution

    Note that calculation may fail if data is too large, due to limitations also imposed by PyOpenCL and ReiknaFFT.
    If this is a problem, you should use the block convolution algorithms below.
    '''
    def __init__(self, shape):
        self.shape = shape

        self.api = cluda.ocl_api()
        self.thr = self.api.Thread.create()

        ocldevice = self.api.get_platforms()[0].get_devices()[0] #Get first device available
        devparam = self.api.DeviceParameters(ocldevice)
        self.maxsize = devparam.max_work_item_sizes

        #check shape is not too large
        if np.product(np.array(shape)) > np.product(np.array(self.maxsize)):
            logging.error(f"Shape {shape} is too large for OpenCL device shape limits {self.maxsize}")
            raise ValueError("Shape is too large.")

        dtype=np.complex64 #Reikna fft only works with complex types
        #Multiply kernel
        multprogram = self.thr.compile("""
        KERNEL void multiply_3d_cplx_arrays(
            GLOBAL_MEM ${ctype} *a,
            GLOBAL_MEM ${ctype} *b,
            GLOBAL_MEM ${ctype} *dest)
        {
            const SIZE_T id0 = get_global_id(0); 
            const SIZE_T id1 = get_global_id(1);
            const SIZE_T id2 = get_global_id(2);

            int IDX = (id0*get_global_size(1)+id1)*get_global_size(2) +id2;

            //Calculate the product value
            dest[IDX] = ${mul}(a[IDX], b[IDX]); //not sure this multiplication is working correctly

        }
        """, render_kwds=dict(
            ctype=dtypes.ctype(dtype),
            mul=functions.mul(dtype, dtype)))
        #Gets a reference to the kernel
        self.kmultiply_3d_cplx_arrays = multprogram.multiply_3d_cplx_arrays

        #Division kernel a/b
        #Ignores complex part and set complex part to zero
        #Also contains a way to prevent division by zero by setting denominator to 1e-14
        divprogram = self.thr.compile("""
        KERNEL void divide_3d_cplx_arrays_realonly(
            GLOBAL_MEM ${ctype} *a,
            GLOBAL_MEM ${ctype} *b,
            GLOBAL_MEM ${ctype} *dest)
        {
            const SIZE_T id0 = get_global_id(0); 
            const SIZE_T id1 = get_global_id(1);
            const SIZE_T id2 = get_global_id(2);

            int IDX = (id0*get_global_size(1)+id1)*get_global_size(2) +id2;

            float a0 = a[IDX].x;
            float b0= b[IDX].x;

            if (b0 ==  0.0) {
                b0= 1e-14;
            };

            //Calculate the division value
            //dest[IDX] = ${div}(a[IDX], b[IDX]);
            //dest[IDX] = ${div}(a[IDX], b0);
            dest[IDX].x = a0/b0;
            dest[IDX].y = 0;

        }
        """, render_kwds=dict(
            ctype=dtypes.ctype(dtype),
            div=functions.div(dtype, dtype)))
        self.kdivide_3d_cplx_arrays_realonly = divprogram.divide_3d_cplx_arrays_realonly

        #Prepare psf fft data in device
        self.psf_fft_dev = self.thr.array(shape, dtype)
        self.psf_flip_fft_dev = self.thr.array(shape, dtype)
        self.is_psf_set = False

        #Prepare the fft kernel using the psf_fft_dev as reference
        self.cfft_gpu = FFT(self.psf_fft_dev).compile(self.thr)


    def setPSF(self,psfdata):
        #Prepare data
        #Convert to float, normalise to sum, calculate fft's and of flipped version
        logging.info("setPSF()")

        psf_norm = convertToFloat32AndNormalise(psfdata , normaliseType='sum', bResetZero=False) #Normalise to sum
        psf0 = change3DSizeTo(psf_norm, self.shape) #Adjust size
        psf1 = circulify3D(psf0) #circulify psf
        psf_cplx = psf1.astype(np.complex64)

        #precalculate psf ffts
        psf_dev = self.thr.to_device(psf_cplx)
        self.cfft_gpu(self.psf_fft_dev , psf_dev) #fft calculation, stores in self.psf_fft_dev
        del(psf1)
        del(psf_dev) #Clear device memory
        #WARNING: If psf is too large, it will throw an error that cannot be caught.
        #

        psf_cplx_flip = np.array(np.flip(psf_cplx))
        psf_flip_dev = self.thr.to_device(psf_cplx_flip)
        #psf_flip_fft_dev = thr.array(shape, np.complex64)
        self.cfft_gpu(self.psf_flip_fft_dev , psf_flip_dev) #fft calculation, stores in self.psf_flip_fft_dev
        del(psf_cplx_flip)
        del(psf_flip_dev)

        self.is_psf_set = True
    
    def doRLDeconvolution(self, data_np, *, niter = 10, callbkTickFunc = None):
        '''
        Does the Richardson Lucy Deconvolution using Reikna OpenCL
        with custom mulitply and division kernels
        
        returns None if fail
        returns the Deconvoluted data volume, only the real part (float32 format)
        '''
        
        logging.info("doRLDeconvolution()")

        #Check data and shape are consistent
        if data_np.shape[0] != self.shape[0] or data_np.shape[1] != self.shape[1] or data_np.shape[2] != self.shape[2]:
            logging.error("Data input shape is different from the shape that was initially set. Please make sure shapes are the same.")
            return None
        
        #Check psf was set
        if self.is_psf_set:

            #Check data. data must be float type
            #No need to check because it will converted to complex anyway
            if not data_np.dtype is np.dtype(np.float32):
                print ("data_np is not np.float32 type. Exiting")
                return None

            #data0 = data_np
            data_cplx = data_np.astype(np.complex64)
            #Send data to device
            data_cplx_dev = self.thr.to_device(data_cplx)

            #ready for the RL iterations

            #Prepare tempary storage on device
            xn1 =  np.array(data_cplx) #initialise copy
            xn1_dev = self.thr.to_device(xn1)
            xn_buff = self.thr.empty_like(xn1_dev)

            t0_dev = self.thr.empty_like(xn1_dev)
            t1_dev = self.thr.empty_like(xn1_dev)

            for i in range(niter):
                xn_dev = xn1_dev

                #Convolution psf*xn
                self.cfft_gpu(t0_dev, xn_dev) # FFT of xn
                self.kmultiply_3d_cplx_arrays( t0_dev , self.psf_fft_dev, t1_dev , local_size=(1,1,1), global_size=self.shape )
                self.cfft_gpu(t0_dev, t1_dev, 1) # FFT inverse, result to t0_dev

                #Division:  image / (psf*xn) , result in t1_dev
                self.kdivide_3d_cplx_arrays_realonly( data_cplx_dev , t0_dev , t1_dev , local_size=(1,1,1), global_size=self.shape )

                #Second convolution with psf_flipped
                self.cfft_gpu(t0_dev, t1_dev) # FFT of t1_dev, result to t0_dev
                self.kmultiply_3d_cplx_arrays( self.psf_flip_fft_dev , t0_dev, t1_dev , local_size=(1,1,1), global_size=self.shape ) #multiplication, result to t1_dev
                self.cfft_gpu(t0_dev, t1_dev, 1) # FFT inverse, result in t0_dev

                #Multiply with xn
                self.kmultiply_3d_cplx_arrays( xn_dev , t0_dev, xn_buff , local_size=(1,1,1), global_size=self.shape ) #multiplication, result to xn_buff

                #Swap buffers
                xn1_dev = xn_buff
                xn_buff = xn_dev

                if not callbkTickFunc is None:
                    callbkTickFunc()

            #Collect result
            xn1 = xn1_dev.get().real
            logging.info("RL completed, result collected")
            
            logging.info("Clearing GPU RAM")
            #Clear GPU memory
            del(xn1_dev)
            del(xn_buff)
            del(t0_dev)
            del(t1_dev)
            #del(psf_flip_fft_dev)
            #del(psf_fft_dev)
            del(data_cplx_dev)

            return xn1

        else:
            logging.error("Psf was not set. Please set it by doing .setPSF(psfdata) .")
            return None
    
    @staticmethod
    def getDefaultDeviceMaxSize():
        api = cluda.ocl_api()

        ocldevice = api.get_platforms()[0].get_devices()[0] #Get first device available
        devparam = api.DeviceParameters(ocldevice)
        maxsize = devparam.max_work_item_sizes
        del(api)
        return maxsize

def nonBlock_RLDeconvolutionReiknaOCL( data_np, psf_np, *, niter = 10, callbkTickFunc=None):
    '''
    This will do the RL deconvolution from 3D data_np using the psf_np provided.
    This does not use block iteration so large arrays may throw out of memory errors

    '''
    logging.info("nonBlock_RLDeconvolutionReiknaOCL()")
    
    if data_np.ndim !=3 or psf_np.ndim!=3:
        logging.error ("Data and psf data must be 3 dimensional. Exiting.")
        return None
    
    data = convertToFloat32AndNormalise(data_np,None,bResetZero=False)
    
    shape = data_np.shape
    myRL_Reikna =  RLDeconv3DReiknaOCL(shape) #Set up calculation by instantiating the class RLDeconv3DReiknaOCL
    myRL_Reikna.setPSF(psf_np)
    
    return myRL_Reikna.doRLDeconvolution(data, niter=niter, callbkTickFunc=callbkTickFunc)

def _isShapeTooBigForDevice(shape):
    ret=False
    
    maxsize = RLDeconv3DReiknaOCL.getDefaultDeviceMaxSize()

    mult_maxsize= np.product(np.array(maxsize))
    mult_shape = np.product(np.array(shape))

    if mult_shape>mult_maxsize:
        ret=True

    return ret


#Default
def block_RLDeconv3DReiknaOCL4(data, psfdata, *, niter=10, max_dim_size=256, psfpaddingfract = 1.2, callbkTickFunc=None):
    '''
    In this version, blockstep is reduced, effectively setting the valid area to a smaller part of the block calculation.
    New parameter psfpaddingfract to set how how much padding relative to psfsize to use
    '''
    logging.info("block_RLDeconv3DReiknaOCL4()")

    if data.ndim !=3 or psfdata.ndim!=3:
        logging.error("Data and psf data must be 3 dimensional. Exiting.")
        raise ValueError("Data or psf are not 3D")
    
    #Check psf size is not too large for the GPU calculation
    if _isShapeTooBigForDevice(psfdata.shape):
        logging.error("PSF data shape is too large. Exiting.")
        raise ValueError("Psf data shape is too large")

    data = convertToFloat32AndNormalise(data)

    shapedata = data.shape
    shapepsf = psfdata.shape

    #print(f"data shape: {shapedata} , psf shape: {shapepsf}")

    #For each of the dimensions.
    #check size is larger than max_dim_size. If it is then limit block calculation using max_dim_size
    blockshape=[0,0,0]
    for a in range(3):
        blockshape0 = max_dim_size #default
        if shapedata[a]<max_dim_size :
            blockshape0 = shapedata[a]
        blockshape[a] = blockshape0
    
    logging.info(f"blockshape: {blockshape}")

    #check if shape is too large
    if _isShapeTooBigForDevice(blockshape):
        logging.error("blockshape is too large. Exiting.")
        raise ValueError("blockshape is too large")

    #Set step (and padding) from the blockshape and padding being psf size *1.5
    #Valid area being the block size minus the psf size *1.5
    blockstep = list(shapedata) #default, makes a copy
    validshape = list(shapedata)
    for a in range(3):
        v0 = int(blockshape[a] - psfpaddingfract*shapepsf[a])
        if blockshape[a]< shapedata[a] and v0>0:
            validshape[a] = v0
            blockstep[a] =  validshape[a]

    logging.info(f"blockstep: {blockstep}")

    datares = np.zeros(data.shape) #To collect results


    #Setup Reikna RL Deconv Class to use this shape
    my_rldeconv = RLDeconv3DReiknaOCL(blockshape)
    my_rldeconv.setPSF(psfdata)

    #do the block iteration, for loops for each dimension
    # the values with zero of the loop are for the left corner
    #The indexes i are for original data
    for iz0 in range(0,shapedata[0], blockstep[0]):
        iz00=iz0
        iz1 = iz0 + blockshape[0]
        if iz1>shapedata[0]:
            iz1 = shapedata[0]
            iz00 = iz1 - blockshape[0]
            if iz00<0: iz00=0
        
        for iy0 in range(0,shapedata[1], blockstep[1]):
            iy00 = iy0
            iy1 = iy0 + blockshape[1]
            if iy1>shapedata[1]:
                iy1 = shapedata[1]
                iy00 = iy1 - blockshape[1]
                if iy00<0: iy00=0

            for ix0 in range(0,shapedata[2], blockstep[2]):
                ix00 = ix0
                ix1 = ix0 + blockshape[2]
                if ix1>shapedata[2]:
                    ix1 = shapedata[2]
                    ix00 = ix1 - blockshape[2]
                    if ix00<0: ix00=0

                logging.info(f"New block, intended origin iz0,iy0,ix0 = {iz0},{iy0},{ix0} , use origin iz00,iy00,ix00 = {iz00},{iy00},{ix00} , end iz1,iy1,ix1 = {iz1},{iy1},{ix1}")

                #Get the data block
                datablock0 = data[iz00:iz1, iy00:iy1, ix00:ix1]
                
                logging.info("Start RL deconvolution of this block")
                #Do RL with this datablock
                rl_of_datablock = my_rldeconv.doRLDeconvolution(datablock0,niter=niter)
                logging.info("This block's RL deconvolution completed")
                
                #Store the datablock result, only the valid part
                #unless it is the leftmost (first block) of the dimension given
                jz0=0
                jy0=0
                jx0=0

                #Crop the padded on the left side
                if iz0 !=0 :
                    jz0 += int( (blockshape[0] - validshape[0]) / 2)
                if iy0 !=0:
                    jy0 += int( (blockshape[1] - validshape[1]) / 2)
                if ix0 !=0:
                    jx0 += int( (blockshape[2] - validshape[2]) / 2)
                
                logging.info(f"Crop block result from origin jz0,jy0,jx0 = : {jz0},{jy0},{jx0}")
                
                logging.info(f"Copying cropped block to datares")
                datares[ iz00+jz0 : iz00+rl_of_datablock.shape[0] , iy00+jy0 : iy00+rl_of_datablock.shape[1] , ix00+jx0 : ix00+rl_of_datablock.shape[2]] = rl_of_datablock[jz0: , jy0: , jx0: ]

                if not callbkTickFunc is None:
                    callbkTickFunc()

    logging.info("Completed block RL deconvolution.")

    #Hopefully, at this point, datares should have the final result of the blocked RL deconvolution
    return datares
    

def printGPUInfo():
    if isReiknaAvailable:
        api = cluda.ocl_api()

        #code from test_reiknaFFT\reikna_getInfo.ipynb
        devices = api.get_platforms()[0].get_devices()
        print(f"OpenCL devices: {devices}")

        for i in range(len(devices)):
            ocldevice = devices[i]
            print(f"Device {i}: {ocldevice}")
            print(f"max_work_group_size: {api.DeviceParameters(ocldevice).max_work_group_size}") # max_work_group_size: 1024
            print(f"max_work_item_sizes: {api.DeviceParameters(ocldevice).max_work_item_sizes}") # max_work_item_sizes: [1024, 1024, 64]
            print(f"max_num_groups: {api.DeviceParameters(ocldevice).max_num_groups}") # max_num_groups: [18446744073709551616, 18446744073709551616, 18446744073709551616]
            print(f"local_mem_size: {api.DeviceParameters(ocldevice).local_mem_size}") # local_mem_size: 49152
            print(f"local_mem_banks: {api.DeviceParameters(ocldevice).local_mem_banks}") # local_mem_banks: 32

        del(api)
    else:
        print("No Reikna.")

