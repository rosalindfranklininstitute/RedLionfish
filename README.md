# Red lionfish (RL) deconvolution

Deconvolution algorithms for busy scientists who would like to do Richardson-Lucy (also RL) deconvolution.

It has a version that runs on CPU (scikit) and another that runs on GPU using OpenCL through Reikna library.

By default it uses the GPU OpenCL version.

If data is too big, it will split into blocks and will run deconvolution in 

If you wish to do RL using GPU openCL on large data, then the 'block' version is automatically done to prevent memory errors.

## Installation

Please note that in order to use OpenCL GPU accelerations, PyopenCL must be installed.
The best way to get it working is to install it under a conda environment.

`conda install reikna pyopencl ocl-icd-system`

Run in the command-line, at the unziped RedLionfish folder:

`python setup.py install`

or download and place in appropriate folder ready to be used. Make sure that it is reachable by python.

If you want to test and modify the code then you should probably install instead using:

`python setup.py develop`

### Install using conda environment

TODO: Under development.

Download the conda package .bzp

## Coding

Please feel free to browse /test folder for examples.

Don't forget to `import RedLionfishDeconv` in order to use the functions:

- `def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False) `


## Build conda package

Execute command-line `conda-build conda-recipe`

or navigate to `conda-recipe`, and execute on the command-line `conda build .`

It will take a while to complete.