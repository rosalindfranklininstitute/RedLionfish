# Red lionfish (RL) deconvolution

Richardson-Lucy deconvolution for fishes, scientists and engineers.

Richardson-Lucy is an iterative deconvolution algorithm that is useful for removing
point spread function (PSF) or optical transfer function artifacts in experimental images.

The method was originally developed for astronomy, to remove optical effects and simultaneously reduce poisson noise in 2D images.

[Lucy, L. B. An iterative technique for the rectification of observed distributions. The Astronomical Journal 79, 745 (1974). DOI: 10.1086/111605](https://ui.adsabs.harvard.edu/abs/1974AJ.....79..745L/abstract)

TODO: More description

It has a version that runs on CPU (scikit) and another that runs on GPU using OpenCL through Reikna library.
By default it uses the GPU OpenCL version.
If data is too big for doing GPU deconvolution, it will do the calculation in blocks, therefore preventing memory issues.


## Installation

It is strongly recommended to install this package under a python anaconda or miniconda environment.
This is because some calculations use PyOpenCL, and this is best installed using `conda install` rather than `pip install`.

It is also possible to install using *pip* and it can still do deconvolution, however the code may fail to use GPU.

It also has a napari plugin. It can be installed by using the napari plugin installer, however this will not enable GPU acceleration.
As above, to best exploit GPU acceleration, it is recommended to install using Anaconda/conda.

### Anaconda/miniconda installation

TODO
`
conda install RedLionfish
`

and it should be ready to use.

If you have installed napari in this conda environment and launch it, the napari-redlionfish plugin should be available in the plugin menu.


#### Manual installation using the conda package.

Download the appropriate conda package .bz2

In the command line, successively run:
```
conda install <filename.bz2>
conda update --all -c conda-forge
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

In your code, add the import.

`import RedLionfishDeconv`

in order to use the functions.

The most useful function is perhaps the following.

`def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False) `

This will do the Richardson-Lucy deconvolution on the data_np (numpy, 3 dimensional data volume) using the provided PSF data volume, for 10 iterations. GPU method is generally faster but it may fail. If it does fail, the program will automatically use the CPU version from scipy.imaging.



## Manually building the conda package

You may need to ensure all packages are installed. For this installation, ensure that the conda-build package is installed

`conda install conda-build`


Then, execute command-line to create the installation package for RedLionfish

`conda-build conda-recipe`

or

`conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe`

Otherwise, navigate to `conda-recipe`, and execute on the command-line `conda build .`

The conda channel conda-forge is important for the installation of reikna and pyopencl, as these are not available in base channels.

It will take a while to complete.