#Testing in local computer using VScode
import sys
sys.path.append('../RedLionfishDeconv')

#from RedLionfishDeconv import *
import RedLionfishDeconv as rl
import tifffile

#Load some data
data = tifffile.imread("test/testdata/gendata_psfconv_poiss_large.tif")
#data = tifffile.imread("test/testdata/gendata_psfconv_poiss.tif")
psf = tifffile.imread("test/testdata/PSF_Rosalind_Casper_8bit.tif")

print(f"data.shape: {data.shape}")
print(f"psf.shape: {psf.shape}")

res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10,resAsUint8=True)
print(f"res_gpu.shape: {res_gpu.shape}")
#res_gpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_gpu)


import napari
np = napari.view_image(res_gpu, ndisplay=3)
napari.run()
