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

Richardson-Lucy (RL) iterative algorithm was developed independently by Richardson [REF] and by Lucy is a well proven and documented computational method for improving experimental images or data, which is widely used in microscopy and astronomy. The image/data recovery requires a known point-spread function (also known as optical transfer function) for executing the mathematical operation of deconvolution.

A pseudo-mathematical description of the image formation and restoration begins by considering a situation where user wants to acquire a perfectly sharp snapshot of an object with a camera/detector. Assuming that the optics are in focus, there is usually instrumental effects and physics limitations that blur the data and add some noised, resulting in a blurred image. Mathematically the blurring process can be described being the result of the original image convoluted with the PSF characteristic of the measuring instrument. The noise is often in the form of Poisson noise but to simplify it will not be discussed here. The *PSF-blurring* , in one dimensional case is given by:

$$MeasuredImage(x,y,z) = \iiint_{-\infty }^{+\infty} ObjectImage(x_0,y_0,z_0) \cdot PSF(x-x_0,y-y_0,z-z_0)dx_0dy_0dz_0 =  ObjectImage \ast PSF $$

or in discrete form
$$MeasuredImage(i,j,k) = \sum_{i_0} \sum_{j_0}\sum_{k_0} ObjectImage_{i_0,j_0,k_0} \cdot PSF_{i,i_0,j,j_0,k,k_0}$$

where $PSF_{i,i_0,j,j_0,k,k_0}$ is usually a function of $i-i_0$ , $j-j_0$, and $k-k_0$

It is very tempting to use the fourier transform's convolution theorem to computationally obtain the ObjectImage from known MeasuredImage and PSF, using the formula $ObjectImage = FFT^{-1}(\frac{FFT(MeasuredImage)}{FFT(PSF)})$ . Although this is mathematically correct, the result obtained is often of very poor quality with amplified noise, which is mainly caused by the division in this formula, meaning that small noises can be greatly amplified.

The Richardson-Lucy iterative algorithm for deconvolution is a way to move from the measured image towards the image without the PSF effect. Being iterative means that the progress can be interrupted and a conveinient step, potentially when the quality is statisfactory and the noise has not reached the point of being gradually amplififed. The mathematical deduction takes a probablistic interpretation of the image data, hence the Bayes theorem can be used to calculate the *inverse* operation.

From Richardson article [REF], and renaming variables here, the one-dimensional form of the restorative iteration in its discrete form is:

$$ EstimateImage_{n+1,i} = EstimateImage_{n,i} \times \sum_{i_0}{ PSF_{i,i_0} \frac{MeasuredImage_{i_0}}{\sum_{i_1}PSF_{i_1,i_0} \cdot EstimateImage_{n,i_1}} }$$

with $n+1$ being the next iteration after $n$.

This formula can be written as convolutions, but it is important to be careful with the PSF indices.

$$ EstimateImage_{n+1} = EstimateImage_{n} \times \left [ { FSP \ast \frac{MeasuredImage}{PSF \ast EstimateImage_{n}} } \right ]$$

with FSP being the *flipped* form of the PSF discrete data, where the value at index 0 becomes the value at the highest index, or, in other words, the data is mirrored along all the three axis. Note that this is not a transpose neither is a rotation. This formula is also valid in the three dimensional case.

DESCRIBE HOW SLOW THIS CALCULATION CAN BE
HIGHLIGHT THE IMPORTANCE OF USING FFT TO DO THE CONVOLUTIONS TO SPEED UP

LARGE VOLUMES, NEED LARGER MEMORY TO DO FFT AND STORE INTERMEDIATE CALCULATIONS

GPUS CAN HELP

CROSS COMPATIBLE, DIFFICULTIES, NOT EVERYONE HAS NVIDIAS

