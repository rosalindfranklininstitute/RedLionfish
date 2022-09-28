#Tests deconvoution of RedLionfish Scipy version (CPU) against others

import numpy as np
import RedLionfishDeconv.RLDeconv3DScipy as rls


def test_doRLDeconvolution_DL2_4_d256_p32():

    data = np.random.random((256,256,256))*10
    psf = np.random.random((16,16,16))

    psf_norm = psf/psf.sum()

    print("Starting RL calculation using doRLDeconvolution_DL2_4()")
    res0 = rls.doRLDeconvolution_DL2_4(data,psf, niter=3)

    print(f"res shape:{res0.shape}, dtype: {res0.dtype}, max,min: {res0.max()},{res0.min()}")

    #Compare with scikit restoration if available
    try:
        import skimage.restoration
    except Exception as e:
        print("No skimage, cannot compare")
        return

    print("Starting RL calculation using skimage.restoration.richardson_lucy()")
    #res_skimage = skimage.restoration.richardson_lucy(data,psf, num_iter = 3, clip=False)
    res_skimage = skimage.restoration.richardson_lucy(data,psf_norm, num_iter = 3, clip=False)

    print(f"res_skimage shape:{res_skimage.shape}, dtype: {res_skimage.dtype}, max,min: {res_skimage.max()},{res_skimage.min()}")

    #comp = np.linalg.norm(res0 - res_skimage) / np.linalg.norm(res0)

    #Compare results of internal areas
    res0_crop = res0[64:192, 64:192,64:192]
    res_skimage_crop = res_skimage[64:192, 64:192,64:192]

    comp = np.linalg.norm(res0_crop - res_skimage_crop) / np.linalg.norm(res0_crop)

    print(f"compare L2 norms of this vs sckit version: {comp}")

    assert comp < 1e-6


