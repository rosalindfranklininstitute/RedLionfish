---
title: 'RedLionfish – Fast Richardson-Lucy Deconvolution package for Efficient PSF Suppression in Volumetric Data'
tags:
  - Python
  - Image processing
  - Microscopy
  - napari
authors:
  - name: Luís M. A. Perdigão
    orcid: 0000-0002-0534-1512
    affiliation: 1
  - name: Casper Berger
    orcid: 0000-0002-0705-3194
    affiliation: 1
  - name: Neville B.-y. Yee
    orcid: 0000-0003-0349-3958
    affiliation: 1
  - name: Michele C. Darrow
    orcid: 0000-0001-6259-1684
    affiliation: 1
  - name: Mark Basham
    orcid: 0000-0002-8438-1415
    affiliation: 1

affiliations:
  - name: The Rosalind Franklin Institute
    index: 1

date: 25 August 2022
bibliography: paper.bib

---
<!-- This is a comment . If you want to comment this article try to look like me -->

# Summary

Experimental limitations in optics in many microscopy and astronomy instruments result in
detrimental effects in imaging of objects, which can be generally described mathematically
as a convolution of the real object image with the point spread function that characterizes
the system.
The popular Richardson-Lucy (RL) deconvolution algorithm is widely used for the inverse process of 
restoring the data without these optical aberrations, which is often a critical restoring step in
data processing of experimental data.
Here we present the versatile RedLionfish python package, that was written to make the RL deconvolution of
volumetric (3D) data easier to run, very fast (by exploiting GPU computing capabilities)
and automatic handling of hardware limitations for large datasets.
It can be used programmatically in Python/numpy using conda or PyPi package managers,
or with a graphical user interace as a napari plugin.

# Statement of need

A pseudo-mathematical description of the image formation and restoration begins by considering a scenario where a user wants to acquire a perfectly sharp snapshot of an object with a camera/detector. Even when the object is in focus, there are additional instrumental effects and physical limitations that blur the data and add noise to the measured image. Mathematically, this blurring process can be described being the result of the original image convoluted with the PSF characteristic of the measuring instrument. The noise is often assumed to be a Poisson noise but it will not discussed here. In three-dimensions, the *PSF-blurring* in the measured image can be modelled by:

$$MI(x,y,z) = \int_{V} OI(x_0,y_0,z_0) \cdot PSF(x-x_0,y-y_0,z-z_0) dx_0dy_0dz_0 =  OI \ast PSF $$

with $MI$ being the Measured Image and $OI$ being the Object Image. In discrete form (pixels or voxels) this can be approximated as:
$$MI_{i,j,k} = \sum_{i_0} \sum_{j_0}\sum_{k_0} OI_{i_0,j_0,k_0} \cdot PSF_{i,i_0,j,j_0,k,k_0}$$

where $PSF_{i,i_0,j,j_0,k,k_0}$ is usually a function of $i-i_0$ , $j-j_0$, and $k-k_0$, which then becomes a convolution.

The question is how to reverse the process, and extract the $OI_{i_0,j_0,k_0}$, with experimentally acquired $MI_{i,j,k}$ and known PSF, a process known as deconvolution. It is tempting to try this by using the fourier transform's convolution theorem, namely using the formula $OI = FFT^{-1}\left(\frac{FFT(MI)}{FFT(PSF)}\right)$ . Although this is mathematically correct, the result obtained is often of poor quality because of the added (unknown) noise in the measured data which is greatly amplified through the division in this formula.
A better solution is to use the Richardson-Lucy iterative algorithm for deconvolution. The Richardson-Lucy (RL) iterative algorithm was developed independently by Richardson [@richardson_bayesian-based_1972], by Lucy [@lucy_iterative_1974] and, equivalently, by others using maximum-likelihood estimation [@shepp_maximum_1982 ; @holmes_richardson-lucymaximum_1989] and is a well proven and documented computational method for improving experimental images or data, which is widely used in microscopy and astronomy [@sarder_deconvolution_2006]. The image/data recovery requires a known point-spread function (PSF, also known as optical transfer function) for executing the mathematical operation of deconvolution. This is a computational algorithm that can be used to suppress the PSF and noise and obtain a good approximation for the object being imaged. Mathematical deduction of the formula takes a probablistic interpretation of the image data, hence the Bayes theorem can be used to calculate the inverse operation in probablistic terms. From Richardson article [@richardson_bayesian-based_1972], and renaming variables here, the one-dimensional form of the restorative iteration in its discrete form is:

$$ EI_{n+1,i} = EI_{n,i} \times \sum_{i_0}{ PSF_{i,i_0} \frac{MI_{i_0}}{\sum_{i_1}PSF_{i_1,i_0} \cdot EI_{n,i_1}} }$$

with $n+1$ being the next iteration after $n$.

This formula can be written as convolutions, but it is important to be careful with the PSF indices meaning that in one of the convolutions the PSF data must be *flipped*.

$$ EI_{n+1} = EI_{n} \times \left [ { FSP \ast \frac{MI}{PSF \ast EI_{n}} } \right ]$$

with FSP being the *flipped* form of the PSF discrete data and $\ast$ representing the multidimensional convolution operation.
This formula is also valid in higher dimensions, maintaining the convolution, PSF flipping and element-wise multiplication. We focus our discussion on the three dimensional case and how to implement and optimize this calculation.

Experimental three-dimensional data is becoming increasingly common
in tomography and light-sheet microscopy being two notable examples.
RL-deconvolution is often an integral part of the data analysis workflow, but it is quite time-consuming and resource hungry in particular when dealing with large data sets.
As such, fast and reliable RL-deconvolution processing can be very useful in accelerating data processing.
We first note that each iteration involves the calculation of two convolutions, one multiplication and one division. The convolution is known of being the slowest since each voxel in the original data is multiplied with each voxel in the PSF, applied to all voxels. The convolution calculation is commonly accelerated using the fast fourier transform (FFT). A single convolution calculation involves three FFT (two forward and one inverse) and a multiplication calculation. This is significantly faster than calculating the convolution by using the sliding PSF method. Despite this algorithmic shortcut, a single iteration of a volume with 1024x1024x64 pixels, and a PSF of about 64x64x64, running 10 iterations can take up to 10 mins.

An additional problem is that the intermediate calculations such as the FFT and other mathematical operations require intermediate storage of the arrays in memory as floating point numbers. With restricted GPU or CPU memory this iterative calculation is likely to throw out-of-memory errors when handling large data volumes. Access to supercomputers may not be the most convenient solution sought for a preprocessing filter.

RedLionfish package was created to address most of these limitations. It is optimised to be fast, by expoiting availability of GPU. It uses PyOpenCL through another package called Reikna which conveniently includes FFT kernels. Since it runs in OpenCL it is also cross-compatible with most CPU's and GPU's, and not restricted to NVIDIA cards. As a failsafe, RedLionfish also has a CPU version of the RL iterative algorithm, optimized for speed. To address potential out-of-memory issues in the GPU calculations, the RL deconvolution can run in blocks (or chunks), removing size limitations of the calculation. To facilitate access to this utility, this package has been made widely available in PyPi, in anaconda environments with condaforge, and as a napari [@noauthor_napari_nodate; @sofroniew_naparinapari_2022] plugin.

![RedLionfish deconvolution of three-dimensional light-microscopy dataset with the sample being plunge frozen HeLa cells on a TEM grid with polysterene beads. (A) is the original data. Spherical beads appear like hourglasses in the experimental image because of instrumental optical astigmatism. In (B) is the result after running the RedLionfish deconvolution for 10 iterations, using a multiple-bead-averaged PSF data. Noticeably the beads appear more spherical and cell appears clearer.\label{fig:fig1}](Figure1.png)
<!-- I removed the kindly provided by, as I am listed as a co-author -->
# Usage

To address different ways that users want to run this deconvolution there are two major options:

- **Napari plugin**: Redlionfish package was made into a napari plugin using the MagicGUI. Input requires the image data, the PSF, number of iterations and optional GPU usage. It is available in pyton package index (PyPI) which makes it easy to install through the napari plugins menu engine, though this is not the recomended installation process, as it may miss PyOpenCL package installation which provide the GPU calculation support. The recommended installation is by using conda, as described in the RedLionfish package webpage. See figure \autoref{fig:fig2} for a screenshot.
- **Programmatically**. By being coded in python, it makes it easy to use programmatically as shown in the example files provided with the package, including Jupyter Notebooks which is excelent for prototyping. Upon package installation in current python environment, it's accessible using the `import RedLionfishDeconv`. The simplest way to run a RL deconvolution on a numpy 3D data array, with a given PSF is by running the function `doRLDeconvolutionFromNpArrays()` with the appropriate parameters. Other specific CPU, GPU or block deconvolution functions are also available. If user requires debugging information in the calculation progress this can enabled by setting `logging.basicConfig(level=logging.INFO)`. The ability to run programmatically also means that it can be included in other packages, such as the correlative image processing software called 3DCT [@arnold_site-specific_2016] (see https://github.com/rosalindfranklininstitute/3DCT).

![Screenshot of RedLionfish user interface in Napari software. \label{fig:fig2}](Figure2.png)

An additional functionality that is included is the ability to monitor calculation progress through a callback method. Full RL-decovolution calculation can take a while to complete and once launched the program may feel like it hangs up. An optional callback method can be passed to the calculation that can be used to reassure the user that a one iteration has completed or, in case of block deconvolution, that it has completed processing one block.

# Implementation

Having the ability to run the speedly RL deconvolution with large datasets, with limited computing resources, can be challenging. It is often desireable to process images while running an experiment to help locate precise 3D position of beads or cells in light microscopy data, which is required before proceeding to the next experimental step. Graphical Processing Units (GPUs) help boost processing speeds but these are often more constrained in memory resources. Redlionfish, by default tries to use the GPU available but if an error occurs it will attempt to process using a *block iterative deconvolution* algorithm, whereby the data is split into smaller volumes and the full RL-deconvolution calculation is run independently for each chunk, and later merged into a single volume [@lee_block-iterative_2015] (\autoref{fig:fig3}). Chunking data is a common technique when dealing with large datasets and is the main reason for the emergence of data processing techniques such as *dask arrays* [REF]. Unfortunately the nature of the RL deconvolution algorithm requires significant amount data from neighbouring volumes which means that in order to get precise results after chunking, padding must be used and edge-effects from each block will need to be taken into account (see below). The block iterative algorithm implemented in RedLionfish has the parameter `psfpaddingfract` (set to 1.2 as default) in function `block_RLDeconv3DReiknaOCL4()` and in file `RLDeconv3DReiknaOCL.py`. This parameter sets how much of the relative size of the PSF data, each of the block edges will be cropped and merged to the final result data volume.

![Simplified schematic of the block algorithm for Richardson-Lucy deconvolution. Data is split into extended blocks. The complete iterative calculation is performed in each block. Then the result is cropped and merged to a single data volume.\label{fig:fig3}](Figure3.png)

It is known in RL-deconvolution that edge errors propagate innwards. Considering the one-dimensional case and data with contiguous number of points of $w_{data}$ and PSF with $w_{PSF}$ data points, and remembering that there are two discrete convolutions in each iteration of the RL algorithm, then the 'valid' region of the reduces by $2w_{PSF}-2$ per iteration. The 'valid' region is defined by being the data range where the calculation of convolution does not relly in padded values at the boundaries, and it uses data points from the image and psf only. This reduction in size is quite significant, for example, imagine that we can split into blocks of data with width of 512 and our PSF is 32. Then a typical 10 iterations would mean that the valid region is reduced to $512-10\times 2(2\times 32 -2) = -108$, less than the original data size meaning that the whole result would be invalid. In this situation, the exact solution in the RL-deconvolution is only possible by running a very low number of iterations, often insuficient to restore data to an restorable level. In fact, in most cases where users use RL-deconvolution to restore data, several iterations are applied, well beyond the limit established for getting positive-sized valid regions. To make matters worse, sometimes users also use size PSF's with the same size as the image data, only to collect the result without bothering to crop the 'valid'-only region. Theoretically the whole result is invalid but despite the reduced precision in using this method, the results obtained are often accepted for further analysis. There are ways that the loss of quality can be mitigated, such as using edge normalisation and block-interlacing methods [@lee_block-iterative_2015], that work well in many cases such as when gaussian PSFs are used and in and with greyscal digital camera photos, however these are not mathematically correct. The reader is advised to check the repository notebooks examples that conduct deeper analysis of the number of iterations and PSF size in reducing the valid region based in a precision criteria. In the blocked processing algorithm, however each block will innevitably display these edge effects, which may or may not be visible, and so the padding fraction may need to be raised to get a better result.

Several solutions to mitigate these rapid reduction of the valid region were carefully considered but none of them was found to be feasible as it would require significantly higher computer processing and memory consumption. This precision issue will be addressed in future versions at expense of speed in case user requires.

<!-- I thought you had some benchmark data obtained on different systems as well? Would that be useful to include? It makes the point of needing a GPU implementation stronger. -->

# Community acceptance

This package was well received by the scientific community from the first day it was published as a napari plugin in PyPI, and shared by other scientists on Twitter and discussed in image.sc forum. There are other free software RL-deconvolution solutions available and users are encouraged to try and compare [@sage_deconvolutionlab2_2017; @lambert_pycudadecon_2022; @noauthor_maweigertgputools_nodate; @haase_clij_2020]. RedLionfish combines many good aspects by being user-friendly through napari, it is reasonably fast, it is free and easily available, and it simply completes the job regardless of your PC.

# Availability

The RedLionfish source code is wrtitten in python is available in github (https://github.com/rosalindfranklininstitute/RedLionfish). It is easily installable in anaconda python environments from conda-forge channel or using pip (PyPI) with the package name being `redlionfish`.

# Acknowledgments

All authors acknowledge funding from Wellcome Trust grants 220526/Z/20/Z and 212980/Z/18/Z.

# References
