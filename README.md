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
This is because some calculations use PyOpenCL, and this is best installed in a conda environment.

```
conda install reikna pyopencl -c conda-forge
pip install redlionfish
```

In Linux , the package `ocl-icd-system` is also useful.

```
conda install reikna pyopencl ocl-icd-system -c conda-forge
pip install redlionfish
```


## Napari plugin

If you follow the installation instructions above, and install the napari in the same conda environment
then the plugin should be immediately available in the *Menu -> Plugins -> RedLionfish*.

Alternatively, you can use the Napari's plugin installation in *Menu -> Plugins -> Install/Uninstall Plugins...*.
If you chose to use this method, GPU acceleration will not be available and it will use the CPU backend.


### Anaconda/miniconda installation

At the moment of this writting this package is NOT AVAILABLE available in conda-forge but is under progress.


#### Manual installation using the conda package file.

Download the appropriate conda package .bz2 at [https://github.com/rosalindfranklininstitute/RedLionfish/releases](https://github.com/rosalindfranklininstitute/RedLionfish/releases)

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

or
`conda install reikna pyopencl ocl-icd-system -c conda-forge` (Linux)

Clone/donload from source [https://github.com/rosalindfranklininstitute/RedLionfish/](https://github.com/rosalindfranklininstitute/RedLionfish/)

and run

`python setup.py install`

### Debug
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

For this installation, ensure that the conda-build package is installed

`conda install conda-build`

In windows, simply execute

`conda-create-package.bat`


Or, execute the command-line to create the installation package for RedLionfish

`conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe`

and the conda package will be created in folder *conda-built-packages*.

Otherwise, navigate to `conda-recipe`, and execute on the command-line `conda build .`

It will take a while to complete.