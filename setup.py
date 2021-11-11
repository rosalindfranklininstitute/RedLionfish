print ("Please note this software can use GPU accelation using PyOpenCL and Reikna.")
print("The Pyopencl package available through pip does not install correctly.")
print("The best way to install PyOpenCL+Reikna  in a python environment is by using anaconda/conda envirnonments.")
print("See more information here: https://documen.tician.de/pyopencl/misc.html")
print("If PyOpenCL+Reikna are not available, it can still run but it will use CPU scipy FFT routines instead which can be slow.")

"""
Copyright (C) 2021 Rosalind Franklin Institute

"""


from setuptools import setup

# import RedLionfishDeconv


setup(
    version = '0.3',
    name = 'RedLionfish',
    description = 'Fast Richardson-Lucy deconvolution of 3D volume data using GPU or CPU.',
    url = 'https://github.com/rosalindfranklininstitute/RedLionfish',
    author = 'Luis Perdigao',
    author_email='luis.perdigao@rfi.ac.uk',
    packages=['RedLionfishDeconv'],
    classifiers=[
        'Development Status :: ok ',
        'License :: Not decided yet',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Windows :: POSIX :: Linux :: MacOS',
    ],
    license='Not sure',
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy'
    ],

    #for the napari plugin
    entry_points={'napari.plugin': 'RedLionfish = RedLionfishDeconv.napari_plugin'}
)
