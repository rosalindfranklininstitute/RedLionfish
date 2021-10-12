# Red lionfish (RL) deconvolution

Deconvolution algorithms for busy scientists who would like to do Richardson-Lucy (also RL) deconvolution.

It has a version that runs on CPU (scikit) and another that runs on GPU using OpenCL through Reikna library.

By default it uses the GPU OpenCL version.

If data is too big, it will split into blocks and will run deconvolution in 

If you wish to do RL using GPU openCL on large data, then the 'block' version is recommended to prevent memory errors.

## Installation

Run:

`python setup.py develop`

or download and place in appropriate folder ready to be used. Make sure that it is reachable by python.


## Coding

Please feel free to browse /test folder for examples.

Don't forget to `import RedLionfishDeconv` in order to use the functions:

- `def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False) `
- or run from command line `python RLDeconvolve.py data.tif psf.tif <iter> --out outfile.tif `

