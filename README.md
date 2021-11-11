# Red lionfish (RL) deconvolution

Deconvolution algorithms for busy scientists who would like to do Richardson-Lucy (also RL) deconvolution.

It has a version that runs on CPU (scikit) and another that runs on GPU using OpenCL through Reikna library.

By default it uses the GPU OpenCL version.

If data is too big, it will split into blocks and will run deconvolution in 

If you wish to do RL using GPU openCL on large data, then the 'block' version is automatically done to prevent memory errors.

## Installation

### Install using conda package, under conda environment (recommended)

Download the appropriate conda package .bz2

In the command line, successively run:
```
conda install <filename.bz2>
conda update --all
```
The second line is needed because you are installing from a local file, conda installer will not install dependencies. Right after this you should run the update command given.

### Manual installation (advanced and for developers)

Please note that in order to use OpenCL GPU accelerations, PyopenCL must be installed.
The best way to get it working is to install it under a conda environment.

`conda install reikna pyopencl`

On linux , the package `ocl-icd-system` is also useful.

Run in the command-line, at the unziped RedLionfish folder:

`python setup.py install`

or download and place in appropriate folder ready to be used. Make sure that it is reachable by python.

If you want to test and modify the code then you should probably install instead using:

`python setup.py develop`


## Coding

Please feel free to browse /test folder for examples.

Don't forget to add

`import RedLionfishDeconv`

in order to use the functions.

The most useful is perhaps the following.

`def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False) `

This will do the Richardson-Lucy deconvolution on the data_np (numpy, 3 dimensional data volume) using the provided PSF data volume, for 10 iterations. GPU method is generally faster but it may fail. If it does fail, the program will automatically use the CPU version from scipy.imaging.



## Manually building conda package

You may need to ensure all packages are installed. For this installation, ensure that the conda-build package is installed

`conda install conda-build`


Then, execute command-line to create the installation package for RedLionfish

`conda-build conda-recipe`

or

`conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe`

Otherwise, navigate to `conda-recipe`, and execute on the command-line `conda build .`

The conda channel conda-forge is important for the installation of reikna and pyopencl, as these are not available in base channels.

It will take a while to complete.