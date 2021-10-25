print ("Please note this software requires pyopencl to work and")
print("the Pyopencl package available through pip does not install correctly.")
print("The best way to install pyopencl in a python environment is by using anaconda/conda.")
print("See more information here: https://documen.tician.de/pyopencl/misc.html")


"""
Copyright (C) 2021 Rosalind Franklin Institute

"""


from setuptools import setup, find_packages

# import RedLionfishDeconv


setup(
    version = '0.1',
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
        'scipy',
        'reikna'
    ]
)
