![RedLionfish Logo](./redlionfish_logo.svg)

# RedLionfish (RL) deconvolution

*Richardson-Lucy deconvolution for fishes, scientists and engineers.*


This software is for filtering 3D data using the Richardson-Lucy deconvolution algorithm.

Richardson-Lucy is an iterative deconvolution algorithm that is used to remove
point spread function (PSF) or optical transfer function (OTF) artifacts from experimental images.

The method was originally developed for astronomy to remove optical effects and simultaneously reduce poisson noise in 2D images.

[Lucy, L. B. An iterative technique for the rectification of observed distributions. The Astronomical Journal 79, 745 (1974). DOI: 10.1086/111605](https://ui.adsabs.harvard.edu/abs/1974AJ.....79..745L/abstract)

The method can also be applied to 3D data. Nowadays this filtering technique is also widely used by microscopists.

The Richardson-Lucy deconvolution algorigthm is iterative. Each iteration involves the calculation of 2 convolutions, one element-wise multiplication and one element-wise division.

When dealing with 3D data, the Richardson-Lucy algorithm is quite computional intensive primarly due to the calculation of the convolution, and can take a while to complete depending on the resources available. Convolution is significantly sped up using FFT compared to raw convolution.

This software was developed with the aim to make the R-L computation faster by exploiting GPU resources, and with the use of FFT convolution.

To make RedLionfish easily accessible, it is available through PyPi and anaconda (conda-forge channel). A useful plugin for Napari is also available.

Please note that this software only works with 3D data. For 2D data there are many alternatives such as the DeconvolutionLab2 in Fiji (ImageJ) and sckikit-image.

## Napari plugin

You can now use the Napari's plugin installation in *Menu -> Plugins -> Install/Uninstall Plugins...*.
However, if you chose to use this method, GPU acceleration may not be available and it will use the CPU backend. Better check.

![](resources\imag1.jpg)

Alternatively, if you follow the installation instructions below, and install the napari in the same python environment
then the plugin should be immediately available in the *Menu -> Plugins -> RedLionfish*.


## Installation

Previously there was a problem in installing using `pip`, because no PyOpenCL wheels for windows were avaiable. It is now avaialble.

This package can be installed using pip or conda.

Napari plugin installation engine can also be used to install this package.


### Install from PyPi

```
pip install redlionfish
```


### Conda install

This package is available in conda-forge channel.
It contains the precompiled libraries and it will install all the requirments for GPU-accelerated RL calculations.

`conda install redlionfish -c conda-forge`

In Linux , the package `ocl-icd-system` may also be useful.

```
conda install reikna pyopencl ocl-icd-system -c conda-forge
```


#### Manual installation using the conda package file.

Download the appropriate conda package .bz2 at [https://github.com/rosalindfranklininstitute/RedLionfish/releases](https://github.com/rosalindfranklininstitute/RedLionfish/releases)

In the command line, successively run:
```
conda install <filename.bz2>
conda update --all -c conda-forge
```
The second line is needed because you are installing from a local file, conda installer will not install dependencies. Right after this you should run the update command given.


### Manual installation (advanced and for developers)

Please note that in order to use OpenCL GPU accelerations, PyOpenCL must be installed.
The best way to get it working is to install it under a conda environment.

The installation is similar to the previously described for PyPi.

`conda install reikna pyopencl`

or

`conda install reikna pyopencl ocl-icd-system -c conda-forge` (Linux)

Clone/download from source [https://github.com/rosalindfranklininstitute/RedLionfish/](https://github.com/rosalindfranklininstitute/RedLionfish/)

and run

`python setup.py install`


### Debug installation
If you want to test and modify the code then you should probably install in debug mode using:

`python setup.py develop`

or

`pip install -e .`


## More information

The software has algorithms for Richardson-Lucy deconvolution that use either CPU and GPU.

The CPU version is very similar to the [skimage.restoration.richardson_lucy](https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.richardson_lucy) code, with improvments in speed.
major differences are:

- the convolution steps use FFT only.
- PSF and PSF-flipped FFTs are precalculated before starting iterations.

The GPU version, was written in to use Reikna package, which does FFT using OpenCL, via PyOpenCL.

Unfortunately, a major limitation in RAM usage exists with PyOpenCL.
Large 3D data volumes with cause out-of-memory error when trying to upload data to the GPU for FFT calculations.
As such, to overcome this problem, a block algorithm is used, which splits data into blocks with padded data.
The results are then combined together to give the final result.
This affects the perfomance of the calculation rather significantly, but with the advantage of being possible to handle large data volumes.

If Richardson-Lucy deconvolution using the GPU method fails, RedLionfish will fallback to CPU calculation. Check console output for messages.

If you are using the RedLionfish in your code, note that, by default, `def doRLDeconvolutionFromNpArrays()` method it uses the GPU OpenCL version.

## Testing

Use pytest to test the package. Test files are in `/test` folder

Many examples can be found in `/scripts' folder.

A useful way to test and benchmark the package installation can be run from the proect root using the command:

'python scripts/test_and_benchm.py'

or in windows

'python scripts\test_and_benchm.py'

This will print out information about your GPU device (if available) and run some deconvolutions.
It initially creates some data programatically, convolutes with a gaussian PSF, and add Poisson noise.
Then it executes executes the
Richardson-Lucy deconvolution calculation using CPU and GPU methods, for 10 iterations.
During the calculation it will print some information to the console/terminal, including the time it takes to run the calculation.


Computer generated data and an experimental PSF can be found in `scripts\testdata`

### Testing Redlionfish in napari

Here is an example testing the Redlionfish plugin in napari:

1. load data `scripts/testdata/gendata_psfconv_poiss_large.tif` (can use draga and drop)
2. load psf data `scripts/testdata/PSF_RFI_8bit.tif`
3. In the RedLionfish side window ensure that 'gendata_psfconv_poiss_large' is selected in data dropdown widget, and `PSF_RFI_8bit` is selected in psfdata widget.
4. Choose number of iterations (default=10)
5. Click 'Go' button and wait until result shows as a new data layer.
6. Use controls of the left panel to compare before and after RL deconvolution: select 'RL-deconvolution' layer and set colormap to red. Hide PSF_RFI_8bit. Make sure that both 'RL-deconvolution' and 'gendata-psfconv' are visible. Now, hide/unhide RL-deconvolution layer to see before and after deconvolution. Adjust contrast limits of each layer as desired.


## GPU vs CPU

You may notice that choosing GPU does not make RL-calculation much faster compared with CPU, and sometimes is slower.

Which method runs the R-L deconvolution faster. That depends on the computer configuration/architecture.

GPU calculations will be generally faster than CPU with bigger data volumes.

GPU calculation will be significantly faster if using a dedicated GPU card.

Please see benchmark values that highlights significant variability in calculation speeds.


[benchmark_results.md](benchmark_results.md)


## Coding

Please feel free to browse the `/scripts` folder for examples.

In order to use the functions, add the follwoing import to your code,

`import RedLionfishDeconv`

The most useful function is perhaps the following.

`def doRLDeconvolutionFromNpArrays(data_np , psf_np ,*, niter=10, method='gpu', useBlockAlgorithm=False, callbkTickFunc=None, resAsUint8 = False) `

This will do the Richardson-Lucy deconvolution on the data_np (numpy, 3 dimensional data volume) using the provided PSF data volume, for 10 iterations. GPU method is generally faster but it may fail. If it does fail, the program will automatically use the CPU version that uses the scipy fft package.



## Manually building the conda package

For this installation, ensure that the conda-build package is installed

`conda install conda-build`

In windows, simply execute

`conda-create-package.bat`


Or, execute the following command-line to create the installation package.

`conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe`

and the conda package will be created in folder *conda-built-packages*.

Otherwise, navigate to `conda-recipe`, and execute on the command-line `conda build .`

It will take a while to complete.

## Contact

Report issues and questions in project's github page, please. Please don't try to send emails as they may be igored or spam-filtered.

