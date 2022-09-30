
#from RedLionfishDeconv import *
import RedLionfishDeconv as rl
import tifffile

import logging
logging.basicConfig(level=logging.INFO)

#Load some data
#data = tifffile.imread("scripts/testdata/gendata_psfconv_poiss_large.tif")
data = tifffile.imread("scripts/testdata/gendata_psfconv_poiss.tif")
psf = tifffile.imread("scripts/testdata/PSF_RFI_8bit.tif")

print(f"data.shape: {data.shape}")
print(f"psf.shape: {psf.shape}")

#test
res_cpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10, method='cpu')
res_cpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_cpu)


res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10)
print(f"res_gpu.shape: {res_gpu.shape}")
res_gpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_gpu)

print("skimage.restoration.richardson_lucy")
#Compare with skimage
import skimage.restoration
res_skimage = skimage.restoration.richardson_lucy(data, psf, iterations=10,clip=False)
print("Completed skimage.restoration.richardson_lucy.")
print(f"res_sk.shape: {res_skimage.shape}")
res_skimage_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_skimage)

import napari
nv = napari.view_image(data, ndisplay=3)
nv.add_image(res_skimage_uint8)
nv.add_image(res_gpu_uint8)
nv.add_image(res_cpu_uint8)

napari.run()
