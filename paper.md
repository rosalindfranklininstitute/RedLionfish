---
title: 'RedLionfish – Fast Richardson-Lucy Deconvolution for Efficient PSF Removal in Volumetric Data'
tags:
  - Python
  - Image processing
  - PSF
  - Microscopy
authors:
  - name: Luís M. A. Perdigão
    orcid: 0000-0002-0534-1512
    affiliation: 1
affiliations:
  - name: The Rosalind Franklin Institute
    index: 1

date: 05 August 2020
bibliography: paper.bib

---

# Summary
Experimental limitations in optics in many microscopy and astronomy instruments result in
detrimental effects in imaging of objects, which can be generally described mathematically
as a convolution of the real object image with the point spread function that characterizes
the system.
The very popular Richardson-Lucy deconvolution algorithm is widely used for the inverse process,
to restore the data without these optical effects.
Here we present the RedLionfish python package that was written to make the deconvolution of
volumetric (3D) data easier to use, very fast (by exploiting GPU computing capabilities)
and automatic handling of hardware limitations for large datasets.
It can also be used programmatically in Python/numpy using conda or PyPi package managers,
and is also a simple napari plugin.


# Statement of need

The Richardson-Lucy (RL) iterative algorithm was developed independently by Richardson [REF] and by Lucy is a well proven and documented computational method for improving experimental images or data, which is widely used in microscopy and astronomy. The image/data recovery requires a known point-spread function (PSF, also known as optical transfer function) for executing the mathematical operation of deconvolution.
 A pseudo-mathematical description of the image formation and restoration begins by considering a situation where user wants to acquire a perfectly sharp snapshot of an object with a camera/detector. Even when optical focus is optimised, there are additional instrumental effects and physical limitations that blur the data and add noise to the measured image. Mathematically, this blurring process can be described being the result of the original image convoluted with the PSF characteristic of the measuring instrument, while the noise is often assumed to be a Poisson noise but it will not discussed. In three-dimensions, the *PSF-blurring* in the measured image can be modelled by:

$$MI(x,y,z) = \int_{V} OI(x_0,y_0,z_0) \cdot PSF(x-x_0,y-y_0,z-z_0) dx_0dy_0dz_0 =  OI \ast PSF $$

with $MI$ being the Measured Image and $OI$ being the Object Image. In discrete form (pixels or voxels) this can be approximated as:
$$MI_{i,j,k} = \sum_{i_0} \sum_{j_0}\sum_{k_0} OI_{i_0,j_0,k_0} \cdot PSF_{i,i_0,j,j_0,k,k_0}$$

where $PSF_{i,i_0,j,j_0,k,k_0}$ is usually a function of $i-i_0$ , $j-j_0$, and $k-k_0$, which then becomes a convolution.

The question is how to reverse the process, and extract the $OI_{i_0,j_0,k_0}$, with experimentally acquired $MI_{i,j,k}$ and known PSF, a process known as *deconvolution*. It is very tempting to try this by using the fourier transform's convolution theorem, namely using the formula $OI = FFT^{-1}\left(\frac{FFT(MI)}{FFT(PSF)}\right)$ . Although this is mathematically correct, the result obtained is often of very poor quality because of the added (unknown) noise in the measured data which is greatly amplified through the division in this formula.

Alternatively, the Richardson-Lucy iterative algorithm for deconvolution is a computational process that can be used to suppress the PSF and noise and obtain a good approximation for the object being imaged. The mathematical deduction takes a probablistic interpretation of the image data, hence the Bayes theorem can be used to calculate the *inverse* operation in probablistic terms. From Richardson article [REF], and renaming variables here, the one-dimensional form of the restorative iteration in its discrete form is:

$$ EI_{n+1,i} = EI_{n,i} \times \sum_{i_0}{ PSF_{i,i_0} \frac{MI_{i_0}}{\sum_{i_1}PSF_{i_1,i_0} \cdot EI_{n,i_1}} }$$

with $n+1$ being the next iteration after $n$.

This formula can be written as convolutions, but it is important to be careful with the PSF indices meaning that in one of the convolutions the PSF data must be *flipped*.

$$ EI_{n+1} = EI_{n} \times \left [ { FSP \ast \frac{MI}{PSF \ast EI_{n}} } \right ]$$

with FSP being the *flipped* form of the PSF discrete data.
This formula is also valid in higher dimensions, maintaining the convolution, PSF flipping and element-wise multiplication. We now focus our discussion here the three dimensional case and how to implement and optimize this calculation.

We first note that each iteration involves the calculation of two convolutions, one multiplication and one division, with the convolution known of being the slowest.
in the two-dimensional case, personal computers nowadays handle these calculations relatively fast, with images with size 1024x1024 taking less than a second per iteration, depending in the computer speed, however three-dimensional data can take significantly longer.
Experimental three-dimensional data is becoming increasingly common
in tomography and light-sheet microscopy being two notable examples.
RL-deconvolution is necessary integral part of the data analysis workflow, but is also quite time-consuming as resource hungry.
As such, fast and reliable RL-deconvolution processing can be very useful.
The convolution calculation is commonly accelerated using the fast fourier transform (FFT). A single convolution calculation involves three FFT (two forward and one inverse) and a multiplication calculation. This is significantly faster than calculating the convolution by suing the sliding PSF method. Despite this algorithmic shortcut, a single iteration of a data volume with 1024x1024x64 pixels, and a PSF of about 64x64x64, running 10 iterations can take up to 10mins.

An additional problem is that the intermediate calculations such as the FFT and other mathematical operations require intermediate storage of the arrays in memory as floating point numbers. With restricted GPU or CPU memory this iterative calculation is likely to throw out-of-memory errors. Access to supercomputers may not be very the most convenient solution sought for a preprocessing filter.

RedLionfish package was created to address most of these difficulties. It is optimised to be fast, by expoiting availability of GPU. It uses PyOpenCL through another package called Reikna which conveniently includes FFT kernels. Since it runs in OpenCL it is also cross-compatible with most CPU's and GPU's, and not restricted to NVIDIA cards. As a failsafe, RedLionfish has also a CPU version of the RL iterative algorithm which is also optimized for speed. To address potential out-of-memory issues in the GPU calculations, the RL deconvolution can run in blocks (or chunks), removing size limitations of the calculation. To facilitate access to this utility, this package has been made widely available in PyPi, in anaconda environments with condaforge, and is a napari [REF] plugin.


