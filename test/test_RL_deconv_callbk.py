'''
This example shows how to do RL deconvolution using either CPU or GPU,
using callback functionality that gives an update of progress of calculation.
'''
#Testing in local computer using VScode, without having installed RedLionfish package
#import sys
#sys.path.append('..')

#from RedLionfishDeconv import *
import RedLionfishDeconv as rl
import tifffile

#Load some data
#data = tifffile.imread("test/testdata/gendata_psfconv_poiss_large.tif")
data = tifffile.imread("test/testdata/gendata_psfconv_poiss.tif")
psf = tifffile.imread("test/testdata/PSF_RFI_8bit.tif")

print(f"data.shape: {data.shape}")
print(f"psf.shape: {psf.shape}")

itick=0
def tickCallBackFunction():
    global itick
    print (f"itick: {itick}")
    itick+=1


print("RL Deconvolution: CPU")
res_cpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10, method='cpu',callbkTickFunc=tickCallBackFunction)
#res_cpu = rl.doRLDeconvolutionFromNpArrays(data,psf,niter=10, method='cpu')
print(f"res_cpu.shape: {res_cpu.shape}")
res_cpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_cpu)

itick=0
print("RL Deconvolution: GPU")
#res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, 10)
res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10, callbkTickFunc=tickCallBackFunction)
print(f"res_gpu.shape: {res_gpu.shape}")
res_gpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_gpu)

#View result with napari
import napari
np = napari.view_image(res_gpu_uint8, ndisplay=3)
np.add_image(res_cpu_uint8)
napari.run()
