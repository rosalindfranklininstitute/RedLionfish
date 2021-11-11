'''
Copyright (C) 2021 Rosalind Franklin Institute

'''


from setuptools import setup, find_packages

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
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Windows :: POSIX :: Linux :: MacOS',
    ],
    license='Apache License, Version 2.0',
    zip_safe=False,
    install_requires=[
        
    ] ,

    #for the napari plugin
    entry_points={'napari.plugin': 'RedLionfish = RedLionfishDeconv.napari_plugin'}
)
