# Calculates RL deconvolution using the scipy, OCL and blockOCL
# and compares consistency between data

# uses fixtures in conftest.py

import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlocl
import RedLionfishDeconv.RLDeconv3DScipy as rlsp
import numpy as np

def test_consistency_with_cubed_data(data_with_cubes_256, psf_gauss_w32_s5):

    print("test_consistency_with_cubed_data()")

    niter0 = 5

    res_scipy = rlsp.doRLDeconvolution(data_with_cubes_256,psf_gauss_w32_s5, niter=niter0)

    res_nb_ocl = rlocl.nonBlock_RLDeconvolutionReiknaOCL(data_with_cubes_256, psf_gauss_w32_s5, niter=niter0 )

    res_b_ocl = rlocl.block_RLDeconv3DReiknaOCL(data_with_cubes_256, psf_gauss_w32_s5, niter=niter0, max_dim_size=128)

    #print sums
    print(f"sum res_scipy: {np.sum(res_scipy)}")
    print(f"sum res_nb_ocl: {np.sum(res_nb_ocl)}")
    print(f"sum res_b_ocl: {np.sum(res_b_ocl)}")

    #Check consistency of relative differences
    maxabsreldiff_sp_nbocl = np.max(np.abs(res_scipy-res_nb_ocl))/np.sum(res_nb_ocl)
    print(f"maxabsreldiff_sp_nbocl: {maxabsreldiff_sp_nbocl}")

    maxabsreldiff_sp_bocl = np.max(np.abs(res_scipy-res_b_ocl))/np.sum(res_b_ocl)
    print(f"maxabsreldiff_sp_bocl: {maxabsreldiff_sp_bocl}")

    maxabsreldiff_nbocl_bocl = np.max(np.abs(res_nb_ocl-res_b_ocl))/np.sum(res_b_ocl)
    print(f"maxabsreldiff_nbocl_bocl: {maxabsreldiff_nbocl_bocl}")

    assert maxabsreldiff_sp_nbocl < 1e-7
    assert maxabsreldiff_sp_bocl < 1e-7
    assert maxabsreldiff_nbocl_bocl < 1e-7


from scipy.optimize import curve_fit
def gaussian(coords3d, a, sigma):
    #Assume it is centered at zero
    return a*np.exp(-0.5*(coords3d[0]**2 + coords3d[1]**2+ coords3d[2]**2)/ sigma**2 )

def getFitGaussParamsOf3DDataAtCentre(data3d, crop_centre_width=None):
    shape = data3d.shape
    centre = [int(w0/2) for w0 in shape]

    #crop a size of 32x32x32
    #data0 = data3d[ centre[0]-16:centre[0]+16, centre[1]-16:centre[1]+16, centre[2]-16:centre[2]+16]
    #range0=16
    data0=None
    if crop_centre_width is None:
        data0=data3d
        range0=int(data3d.shape[0]/2) #Assumes it is a cube
    else:
        range0 = int(crop_centre_width/2)
        data0 = data3d[ centre[0]-range0:centre[0]+range0, centre[1]-range0:centre[1]+range0, centre[2]-range0:centre[2]+range0]
    xrange=np.arange(-range0,range0)
    yrange = np.array(xrange)
    zrange = np.array(xrange)

    #guess amplitude
    a0=data0.max()
    xx,yy,zz = np.meshgrid(xrange,yrange,zrange)

    xx_flat= xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = zz.flatten()

    #coordsvalues = list(zip(xx_flat, yy_flat, zz_flat))

    coordsvalues = np.array([zz_flat,yy_flat,xx_flat])
    #print(coordsvalues.shape)
    
    data0_flat = data0.flatten()

    popt, pcov = curve_fit(gaussian,coordsvalues, data0_flat, p0=[a0,range0])
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

def test_consistency_in_gaussian_widths(data_gauss_w256_s5, psf_gauss_w32_s5):

    print("test_consistency_in_gaussian_widths()")

    niter0 = 3

    print("Doing RL deconvolutions.")
    res_scipy = rlsp.doRLDeconvolution(data_gauss_w256_s5,psf_gauss_w32_s5, niter=niter0)

    res_nb_ocl = rlocl.nonBlock_RLDeconvolutionReiknaOCL(data_gauss_w256_s5, psf_gauss_w32_s5, niter=niter0 )

    res_b_ocl = rlocl.block_RLDeconv3DReiknaOCL(data_gauss_w256_s5, psf_gauss_w32_s5, niter=niter0, max_dim_size=128)

    print("Fitting gaussians to results.")
    #Fit 3D gaussians
    g_scipy = getFitGaussParamsOf3DDataAtCentre(res_scipy,crop_centre_width=32)
    g_nb_ocl = getFitGaussParamsOf3DDataAtCentre(res_nb_ocl, crop_centre_width=32)
    g_b_ocl = getFitGaussParamsOf3DDataAtCentre(res_b_ocl, crop_centre_width=32)

    w_scipy = g_scipy[0][1]
    w_nb_ocl = g_nb_ocl[0][1]
    w_b_ocl = g_b_ocl[0][1]

    print(f"w_scipy: {w_scipy}")
    print(f"w_nb_ocl: {w_nb_ocl}")
    print(f"w_b_ocl: {w_b_ocl}")

    assert w_scipy<4
    assert w_nb_ocl<4
    assert w_b_ocl<4
