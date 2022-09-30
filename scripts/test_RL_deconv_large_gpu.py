#Testing in local computer using VScode, without having installed RedLionfish package
#import sys
#sys.path.append('..')

#from RedLionfishDeconv import *
import RedLionfishDeconv as rl
import tifffile

#Load some data
data = tifffile.imread("scripts/testdata/gendata_psfconv_poiss_large.tif")
#data = tifffile.imread("scripts/testdata/gendata_psfconv_poiss.tif")
psf = tifffile.imread("scripts/testdata/PSF_RFI_8bit.tif")

print(f"data.shape: {data.shape}")
print(f"psf.shape: {psf.shape}")

res_gpu = rl.doRLDeconvolutionFromNpArrays(data,psf, niter=10,resAsUint8=True)
print(f"res_gpu.shape: {res_gpu.shape}")
#res_gpu_uint8 = rl.helperfunctions.convertToUint8AndFullRange(res_gpu)


import napari
np = napari.view_image(data, ndisplay=3)
np.add_image(res_gpu)
napari.run()
