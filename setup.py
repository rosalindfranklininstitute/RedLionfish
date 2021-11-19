print("Note this software can use GPU accelation using PyOpenCL and Reikna.")
print("Please install these packages seperately.")
print("The PyOpenCL package available through pip may not install correctly and/or provide access to GPU.")
print("The best way to install PyOpenCL+Reikna in a python environment is by using anaconda/conda envirnonments.")
print("See more information here: https://documen.tician.de/pyopencl/misc.html")
print("If PyOpenCL+Reikna packages are not available, RedLionfish can still run but it will use CPU backend of scipy FFT routines, which can be slow.")

'''
Copyright (C) 2021 Rosalind Franklin Institute

'''

from setuptools import setup

setup(
    version = '0.4',
    name = 'RedLionfish',
    description = 'Fast Richardson-Lucy deconvolution of 3D volume data using GPU or CPU with napari plugin.',
    url = 'https://github.com/rosalindfranklininstitute/RedLionfish',
    author = 'Luis Perdigao',
    author_email='luis.perdigao@rfi.ac.uk',
    packages=['RedLionfishDeconv'],
    classifiers=[
        'Development Status :: 4 - Beta ',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Framework :: napari'
    ],
    license='Apache License, Version 2.0',
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy'
    ],

    #for the napari plugin
    entry_points={'napari.plugin': 'RedLionfish = RedLionfishDeconv.napari_plugin'}
)
